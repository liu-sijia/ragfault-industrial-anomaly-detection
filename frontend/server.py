from __future__ import annotations

import csv
import io
import os
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
SRC = ROOT / "src"
EXT = ROOT / "extension_experiment"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXT))

from data_processor import load_config, process_data_files_from_list  # noqa: E402
from evaluator import calculate_metrics  # noqa: E402
from knowledge_base import initialize_knowledge_base  # noqa: E402
from llm_detector import LLMDetector  # noqa: E402
from rag_retriever import RAGRetriever  # noqa: E402
from scene_knowledge_builder import build_scene_knowledge_base, list_scene_knowledge_bases  # noqa: E402
from detectors.generic_llm_detector import GenericLLMDetector  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent))
DEFAULT_KB_DIR = PROJECT / "knowledge_base"
SCENES_ROOT = PROJECT / "knowledge_base" / "scenes"
CONFIG_PATH = Path(os.environ.get("ANOMALY_RAG_CONFIG", PROJECT / "config.example.yaml"))
HAI_CONFIG_PATH = Path(
    os.environ.get(
        "ANOMALY_RAG_HAI_CONFIG",
        EXT / "configs" / "hai_transfer_config.example.yaml",
    )
)
HAI_KB_DIR = EXT / "knowledge_base" / "hai"
HAI_PROMPT_PATH = EXT / "prompts" / "hai_prompt.txt"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/<path:filename>")
def static_file(filename: str):
    return send_from_directory(app.static_folder, filename)


@app.get("/api/knowledge/scenes")
def knowledge_scenes():
    return jsonify({"scenes": list_builtin_and_scene_knowledge_bases()})


@app.post("/api/knowledge/build")
def build_knowledge():
    document = request.files.get("document")
    csv_upload = request.files.get("csv")
    if not document or not csv_upload:
        return jsonify({"error": "请同时上传工程资料文档和用于字段匹配的 CSV。"}), 400

    config = load_config(str(CONFIG_PATH))
    scene_name = (request.form.get("scene_name") or "").strip() or None
    extract_model = (request.form.get("extract_model") or "qwen2.5:7b").strip()
    ollama_url = config.get("llm", {}).get("local_url", "http://localhost:11434")
    embedding_model = config.get("knowledge_base", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")

    with tempfile.TemporaryDirectory(prefix="llm_rag_kb_") as tmpdir:
        tmp = Path(tmpdir)
        doc_path = tmp / secure_document_name(document.filename or "document.txt")
        csv_path = tmp / secure_csv_name(csv_upload.filename or "data.csv")
        document.save(doc_path)
        csv_upload.save(csv_path)

        try:
            scene = build_scene_knowledge_base(
                document_path=doc_path,
                csv_path=csv_path,
                output_root=SCENES_ROOT,
                scene_name=scene_name,
                ollama_url=ollama_url,
                model=extract_model,
                embedding_model=embedding_model,
                build_index=True,
                allow_fallback=True,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return jsonify({"scene": scene, "scenes": list_scene_knowledge_bases(DEFAULT_KB_DIR, SCENES_ROOT)})


@app.post("/api/detect")
def detect():
    uploads = request.files.getlist("files") or request.files.getlist("file")
    if not uploads:
        return jsonify({"error": "请上传 CSV 文件或包含 CSV 的 ZIP。"}), 400

    config = load_config(str(CONFIG_PATH))
    config["data"]["window_size"] = int(request.form.get("window_size", config["data"]["window_size"]))
    config["data"]["window_step"] = config["data"]["window_size"]
    config["data"]["baseline_points"] = int(request.form.get("baseline_points", config["data"].get("baseline_points", 400)))
    config["knowledge_base"]["top_k"] = int(request.form.get("top_k", config["knowledge_base"]["top_k"]))
    scene_id = (request.form.get("scene_id") or "default").strip()
    if scene_id == "hai":
        return detect_hai(uploads)
    try:
        config["knowledge_base"]["dir"] = str(resolve_scene_kb_dir(scene_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with tempfile.TemporaryDirectory(prefix="llm_rag_upload_") as tmpdir:
        csv_files: list[str] = []
        for index, upload in enumerate(uploads):
            filename = upload.filename or f"uploaded_{index}.csv"
            saved_path = Path(tmpdir) / f"{index:03d}_{secure_upload_name(filename)}"
            upload.save(saved_path)
            if saved_path.suffix.lower() == ".zip":
                csv_files.extend(extract_csv_zip(saved_path, Path(tmpdir) / f"zip_{index:03d}"))
            elif saved_path.suffix.lower() == ".csv":
                csv_files.append(str(saved_path))

        if not csv_files:
            return jsonify({"error": "上传内容中没有找到 CSV 文件。"}), 400

        old_cwd = os.getcwd()
        os.chdir(PROJECT)
        try:
            windows, _ = process_data_files_from_list(config, csv_files)
            if not windows:
                return jsonify({"error": "CSV 未生成有效窗口，请检查 datetime、传感器列和数据时长。"}), 400

            kb_info = initialize_knowledge_base(config, rebuild=False)
            retriever = RAGRetriever(config, kb_info)
            detector = LLMDetector(config)

            def process_window(window: dict[str, Any]) -> dict[str, Any]:
                rag_context = retriever.retrieve(window["stats"])
                prediction, explanation, success = detector.detect(window["stats"], rag_context)
                window["prediction"] = int(prediction)
                window["explanation"] = explanation
                window["success"] = bool(success)
                window["rag_context"] = rag_context
                window["evidence"] = parse_evidence(rag_context)
                return window

            processed: list[dict[str, Any]] = []
            max_workers = int(request.form.get("workers", 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_window, w) for w in windows]
                for future in as_completed(futures):
                    processed.append(future.result())
            processed.sort(key=lambda item: item.get("window_idx", 0))
        except Exception as exc:  # keeps the browser error clear during demos
            return jsonify({"error": str(exc)}), 500
        finally:
            os.chdir(old_cwd)

    y_true = [int(w.get("label", 0)) for w in processed]
    y_pred = [int(w.get("prediction", 0)) for w in processed]
    metrics = calculate_metrics(y_true, y_pred)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    result_windows = [serialize_window(w, config["data"]["window_size"]) for w in processed]
    predictions_csv = build_predictions_csv(result_windows)
    actual_metrics = build_metric_payload(metrics, tp, tn, fp, fn, len(result_windows), sum(y_true), sum(y_pred))

    return jsonify(
        {
            "file_name": uploads[0].filename if len(uploads) == 1 else f"{len(csv_files)} CSV files",
            "file_count": len(csv_files),
            "metrics": actual_metrics,
            "windows": result_windows,
            "evidence_summary": summarize_evidence(result_windows),
            "predictions_csv": predictions_csv,
        }
    )


def list_builtin_and_scene_knowledge_bases() -> list[dict[str, Any]]:
    scenes = list_scene_knowledge_bases(DEFAULT_KB_DIR, SCENES_ROOT)
    if HAI_KB_DIR.exists():
        scenes.insert(
            1,
            {
                "id": "hai",
                "name": "HAI迁移知识库",
                "path": str(HAI_KB_DIR.resolve()),
                "builtin": True,
                "sensor_count": 16,
                "anomaly_pattern_count": 6,
                "mode": "hai",
            },
        )
    return scenes


def build_metric_payload(
    metrics: dict[str, Any],
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    total: int,
    anomalies: int,
    predicted: int,
) -> dict[str, Any]:
    return {
        "F1": metrics["F1"],
        "FAR": metrics["FAR"],
        "MAR": metrics["MAR"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Accuracy": metrics["Accuracy"],
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total": total,
        "anomalies": anomalies,
        "predicted": predicted,
    }


def detect_hai(uploads):
    config = load_config(str(HAI_CONFIG_PATH))
    config["data"]["window_size"] = int(request.form.get("window_size", config["data"]["window_size"]))
    config["data"]["window_step"] = config["data"]["window_size"]
    config["data"]["baseline_points"] = int(request.form.get("baseline_points", config["data"].get("baseline_points", 180)))
    config["knowledge_base"]["top_k"] = int(request.form.get("top_k", config["knowledge_base"]["top_k"]))
    config["knowledge_base"]["dir"] = str(HAI_KB_DIR)

    with tempfile.TemporaryDirectory(prefix="llm_rag_hai_upload_") as tmpdir:
        csv_files: list[str] = []
        for index, upload in enumerate(uploads):
            filename = upload.filename or f"uploaded_{index}.csv"
            saved_path = Path(tmpdir) / f"{index:03d}_{secure_upload_name(filename)}"
            upload.save(saved_path)
            if saved_path.suffix.lower() == ".zip":
                csv_files.extend(extract_csv_zip(saved_path, Path(tmpdir) / f"zip_{index:03d}"))
            elif saved_path.suffix.lower() == ".csv":
                csv_files.append(str(saved_path))

        if not csv_files:
            return jsonify({"error": "上传内容中没有找到 CSV 文件。"}), 400

        old_cwd = os.getcwd()
        os.chdir(PROJECT)
        try:
            windows = create_hai_windows_from_files(config, csv_files)
            if not windows:
                return jsonify({"error": "HAI CSV 未生成有效窗口，请检查字段、anomaly/attack 标签和数据长度。"}), 400

            kb_info = initialize_knowledge_base(config, rebuild=False)
            retriever = RAGRetriever(config, kb_info)
            detector = GenericLLMDetector(config, HAI_PROMPT_PATH)

            def process_window(window: dict[str, Any]) -> dict[str, Any]:
                rag_context = retriever.retrieve(window["stats"])
                rule_prediction = configured_hai_rule_predict(window, config)
                try:
                    raw_prediction, explanation, _prompt = detector.detect(window["stats"], rag_context)
                    success = True
                except Exception as exc:
                    raw_prediction = rule_prediction
                    explanation = f"LLM 调用失败，使用 HAI 规则约束判定：{exc}"
                    success = False
                final_mode = config.get("experiment", {}).get("final_constraint_mode", "rag_guard")
                prediction = rule_prediction if final_mode == "rag_guard" else max(raw_prediction, rule_prediction)
                window["prediction"] = int(prediction)
                window["llm_raw_prediction"] = int(raw_prediction)
                window["rule_constraint_prediction"] = int(rule_prediction)
                window["explanation"] = explanation
                window["success"] = success
                window["rag_context"] = rag_context
                window["evidence"] = parse_evidence(rag_context)
                return window

            processed: list[dict[str, Any]] = []
            max_workers = int(request.form.get("workers", 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_window, w) for w in windows]
                for future in as_completed(futures):
                    processed.append(future.result())
            processed.sort(key=lambda item: item.get("window_idx", 0))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        finally:
            os.chdir(old_cwd)

    y_true = [int(w.get("label", 0)) for w in processed]
    y_pred = [int(w.get("prediction", 0)) for w in processed]
    metrics = calculate_metrics(y_true, y_pred)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    result_windows = [serialize_window(w, config["data"]["window_size"]) for w in processed]
    predictions_csv = build_predictions_csv(result_windows)

    return jsonify(
        {
            "file_name": uploads[0].filename if len(uploads) == 1 else f"{len(csv_files)} CSV files",
            "file_count": len(csv_files),
            "mode": "HAI",
            "metrics": {
                "F1": metrics["F1"],
                "FAR": metrics["FAR"],
                "MAR": metrics["MAR"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "Accuracy": metrics["Accuracy"],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "total": len(result_windows),
                "anomalies": sum(y_true),
                "predicted": sum(y_pred),
            },
            "windows": result_windows,
            "evidence_summary": summarize_evidence(result_windows),
            "predictions_csv": predictions_csv,
        }
    )


def create_hai_windows_from_files(config: dict[str, Any], csv_files: list[str]) -> list[dict[str, Any]]:
    features = config["data"]["features"]
    baseline_points = int(config["data"].get("baseline_points", 180))
    window_size = int(config["data"]["window_size"])
    window_step = int(config["data"]["window_step"])
    windows: list[dict[str, Any]] = []

    for file_path in csv_files:
        df = read_flexible_csv(file_path)
        if "time" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"time": "datetime"})
        if "attack" in df.columns and "anomaly" not in df.columns:
            df = df.rename(columns={"attack": "anomaly"})
        if "datetime" not in df.columns:
            raise ValueError("HAI 数据需要包含 datetime 或 time 列。")
        if "anomaly" not in df.columns:
            raise ValueError("HAI 数据需要包含 anomaly 或 attack 标签列。")

        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"HAI 数据缺少字段：{', '.join(missing[:8])}")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).astype(int)
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df = df.dropna(subset=features).reset_index(drop=True)

        segment_values = sorted(df["segment_id"].dropna().unique()) if "segment_id" in df.columns else [None]
        for segment_id in segment_values:
            segment_df = df[df["segment_id"] == segment_id].reset_index(drop=True) if segment_id is not None else df
            if len(segment_df) < window_size:
                continue
            baseline_df = segment_df.head(min(baseline_points, len(segment_df)))
            baseline = {
                feature: {
                    "mean": float(baseline_df[feature].mean()),
                    "std": float(baseline_df[feature].std()),
                }
                for feature in features
            }

            for start in range(0, len(segment_df) - window_size + 1, window_step):
                window_df = segment_df.iloc[start : start + window_size]
                stats = {}
                for feature in features:
                    values = window_df[feature]
                    std = baseline[feature]["std"]
                    z_score = (float(values.mean()) - baseline[feature]["mean"]) / std if std > 1e-9 else 0.0
                    stats[feature] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "max": float(values.max()),
                        "z_score": float(z_score),
                    }
                source_start = int(window_df["source_index"].iloc[0]) if "source_index" in window_df.columns else int(start)
                source_end = int(window_df["source_index"].iloc[-1]) if "source_index" in window_df.columns else int(start + window_size - 1)
                windows.append(
                    {
                        "window_idx": len(windows),
                        "segment_id": int(segment_id) if segment_id is not None else 0,
                        "start": source_start,
                        "end": source_end,
                        "label": int(window_df["anomaly"].max()),
                        "stats": stats,
                    }
                )
    return windows


def read_flexible_csv(file_path: str) -> pd.DataFrame:
    errors: list[str] = []
    for kwargs in ({"sep": None, "engine": "python"}, {"sep": ";"}, {"sep": ","}):
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"无法读取 CSV：{'; '.join(errors[:2])}")


def configured_hai_rule_predict(window: dict[str, Any], config: dict[str, Any]) -> int:
    experiment = config.get("experiment", {})
    severe = float(experiment.get("severe_z_threshold", 5.0))
    multi = float(experiment.get("multi_z_threshold", 2.0))
    multi_count = int(experiment.get("multi_z_count", 2))
    z_scores = [abs(float(stats.get("z_score", 0.0))) for stats in window["stats"].values() if isinstance(stats, dict)]
    return int(max(z_scores, default=0.0) >= severe or sum(z >= multi for z in z_scores) >= multi_count)


def secure_csv_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    return name if name.lower().endswith(".csv") else f"{name}.csv"


def secure_upload_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    if name.lower().endswith((".csv", ".zip")):
        return name
    return f"{name}.csv"


def secure_document_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    if name.lower().endswith((".txt", ".md", ".docx", ".pdf")):
        return name
    return f"{name}.txt"


def resolve_scene_kb_dir(scene_id: str) -> Path:
    if scene_id in {"", "default"}:
        return DEFAULT_KB_DIR
    if scene_id == "hai":
        return HAI_KB_DIR
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", scene_id):
        raise ValueError("知识库标识不合法。")
    scene_dir = SCENES_ROOT / scene_id
    if not scene_dir.exists():
        raise ValueError(f"未找到知识库：{scene_id}")
    return scene_dir


def extract_csv_zip(zip_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            safe_name = secure_csv_name(member.filename.replace("/", "_").replace("\\", "_"))
            target = out_dir / safe_name
            with archive.open(member) as source, target.open("wb") as dest:
                dest.write(source.read())
            csv_files.append(str(target))
    return csv_files


def parse_evidence(rag_context: str) -> list[dict[str, str]]:
    if "=== Domain Knowledge ===" not in rag_context:
        return []
    domain = rag_context.split("=== Domain Knowledge ===", 1)[1].strip()
    matches = re.findall(r"\[\d+\]\s*(.*?)(?=\n\[\d+\]\s*|\Z)", domain, re.S)
    evidence = []
    for text in matches:
        clean = " ".join(text.split())
        if not clean:
            continue
        title = clean.split(" - ", 1)[0][:40]
        evidence.append({"title": title, "text": clean[:500]})
    return evidence


def serialize_window(window: dict[str, Any], window_size: int) -> dict[str, Any]:
    idx = int(window.get("window_idx", 0))
    stats = {}
    max_abs_z = 0.0
    for feature, values in window["stats"].items():
        if not isinstance(values, dict):
            continue
        z = float(values.get("z_score", 0.0))
        max_abs_z = max(max_abs_z, abs(z))
        stats[feature] = {
            "mean": float(values.get("mean", 0.0)),
            "std": float(values.get("std", 0.0)),
            "max": float(values.get("max", 0.0)),
            "z": z,
            "z_score": z,
        }
    return {
        "id": idx,
        "window_idx": idx,
        "start": int(window.get("start", idx * window_size)),
        "end": int(window.get("end", idx * window_size + window_size - 1)),
        "label": int(window.get("label", 0)),
        "prediction": int(window.get("prediction", 0)),
        "reason": window.get("explanation", ""),
        "success": bool(window.get("success", True)),
        "stats": stats,
        "maxAbsZ": max_abs_z,
        "evidence": window.get("evidence", []),
    }


def summarize_evidence(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for window in windows:
        for item in window.get("evidence", []):
            title = item.get("title", "RAG evidence")
            if title not in counts:
                counts[title] = {"title": title, "text": item.get("text", ""), "count": 0}
            counts[title]["count"] += 1
    return sorted(counts.values(), key=lambda item: item["count"], reverse=True)


def build_predictions_csv(windows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["window_idx", "label", "prediction", "max_abs_z", "llm_reason", "rag_evidence"])
    for window in windows:
        writer.writerow(
            [
                window["window_idx"],
                window["label"],
                window["prediction"],
                f"{window['maxAbsZ']:.6f}",
                window["reason"],
                "; ".join(item.get("title", "") for item in window.get("evidence", [])[:3]),
            ]
        )
    return output.getvalue()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
