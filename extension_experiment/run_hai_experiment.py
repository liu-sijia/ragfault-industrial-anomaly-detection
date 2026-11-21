from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXT = ROOT / "extension_experiment"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXT))

from adapters.hai_adapter import build_hai_subset  # noqa: E402
from detectors.generic_llm_detector import GenericLLMDetector  # noqa: E402
from evaluator import calculate_metrics  # noqa: E402
from knowledge_base import initialize_knowledge_base  # noqa: E402
from rag_retriever import RAGRetriever  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_windows(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["processed_csv"], parse_dates=["datetime"])
    features = data_cfg["features"]
    baseline_points = int(data_cfg["baseline_points"])
    window_size = int(data_cfg["window_size"])
    window_step = int(data_cfg["window_step"])

    windows: list[dict[str, Any]] = []
    segment_values = sorted(df["segment_id"].unique()) if "segment_id" in df.columns else [0]
    for segment_id in segment_values:
        segment_df = df[df["segment_id"] == segment_id].reset_index(drop=True) if "segment_id" in df.columns else df
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
                    "segment_id": int(segment_id),
                    "start": source_start,
                    "end": source_end,
                    "label": int(window_df["anomaly"].max()),
                    "stats": stats,
                }
            )

    max_windows = int(config.get("experiment", {}).get("max_windows", 0) or 0)
    return windows[:max_windows] if max_windows > 0 else windows


def rule_predict(window: dict[str, Any], severe: float, multi: float) -> int:
    z_scores = [abs(float(stats["z_score"])) for stats in window["stats"].values()]
    multi_count = int(window.get("multi_z_count", 2))
    return int(max(z_scores) >= severe or sum(z >= multi for z in z_scores) >= multi_count)


def configured_rule_predict(window: dict[str, Any], config: dict[str, Any]) -> int:
    experiment = config["experiment"]
    enriched_window = {**window, "multi_z_count": int(experiment.get("multi_z_count", 2))}
    return rule_predict(
        enriched_window,
        float(experiment["severe_z_threshold"]),
        float(experiment["multi_z_threshold"]),
    )


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    metrics.update({"TP": tp, "TN": tn, "FP": fp, "FN": fn, "total": len(y_true), "anomalies": sum(y_true)})
    return metrics


def run_llm_group(
    name: str,
    windows: list[dict[str, Any]],
    detector: GenericLLMDetector,
    retriever: RAGRetriever,
    use_rag: bool,
    output_dir: Path,
    constrained_final: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []

    for window in windows:
        rag_context = retriever.retrieve(window["stats"]) if use_rag else "No retrieved domain context."
        raw_prediction, reason, prompt = detector.detect(window["stats"], rag_context)
        constrained_prediction = configured_rule_predict(window, detector.config)
        final_mode = detector.config.get("experiment", {}).get("final_constraint_mode", "max")
        if constrained_final and final_mode == "rag_guard":
            prediction = constrained_prediction
        elif constrained_final:
            prediction = max(raw_prediction, constrained_prediction)
        else:
            prediction = raw_prediction
        predictions.append(prediction)
        evidence_titles = extract_evidence_titles(rag_context)
        max_abs_z = max(abs(float(stats["z_score"])) for stats in window["stats"].values())
        rows.append(
            {
                "window_idx": window["window_idx"],
                "segment_id": window.get("segment_id", 0),
                "start": window["start"],
                "end": window["end"],
                "label": window["label"],
                "prediction": prediction,
                "llm_raw_prediction": raw_prediction,
                "rule_constraint_prediction": constrained_prediction,
                "correct": int(window["label"] == prediction),
                "max_abs_z": f"{max_abs_z:.4f}",
                "reason": reason,
                "rag_evidence": "; ".join(evidence_titles[:3]),
                "prompt": prompt,
            }
        )

    y_true = [int(w["label"]) for w in windows]
    metrics = evaluate_predictions(y_true, predictions)
    write_csv(output_dir / "predictions.csv", rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{name}: F1={metrics['F1']:.4f}, FAR={metrics['FAR']:.4f}, MAR={metrics['MAR']:.4f}")
    return metrics


def extract_evidence_titles(rag_context: str) -> list[str]:
    titles = []
    for line in rag_context.splitlines():
        line = line.strip()
        if line.startswith("["):
            title = line.split(" - ", 1)[0]
            titles.append(title[:80])
    return titles


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated HAI extension experiment.")
    parser.add_argument(
        "--config",
        default=str(EXT / "configs" / "hai_transfer_config.example.yaml"),
    )
    parser.add_argument("--skip-llm", action="store_true", help="Only build data/KB and run baselines.")
    args = parser.parse_args()

    config_path = Path(args.config)
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        config = load_config(config_path)
        build_hai_subset(
            raw_gzip=config["data"]["raw_gzip"],
            output_csv=config["data"]["processed_csv"],
            features=config["data"]["features"],
            source_start=int(config["data"]["source_start"]),
            source_end=int(config["data"]["source_end"]),
            segments=config["data"].get("segments"),
        )
        windows = create_windows(config)
        print(f"HAI subset windows: {len(windows)}, anomaly windows: {sum(w['label'] for w in windows)}")

        severe = float(config["experiment"]["severe_z_threshold"])
        multi = float(config["experiment"]["multi_z_threshold"])
        y_true = [int(w["label"]) for w in windows]
        all_anomaly_preds = [1] * len(windows)
        all_anomaly_metrics = evaluate_predictions(y_true, all_anomaly_preds)
        rule_preds = [configured_rule_predict(w, config) for w in windows]
        rule_metrics = evaluate_predictions(y_true, rule_preds)
        results_dir = Path(config.get("experiment", {}).get("results_dir", EXT / "results" / "hai"))
        results_dir.mkdir(parents=True, exist_ok=True)
        include_zscore_reference = bool(config.get("experiment", {}).get("include_zscore_reference", True))
        (results_dir / "all_anomaly_baseline.json").write_text(json.dumps(all_anomaly_metrics, indent=2), encoding="utf-8")
        (results_dir / "zscore_rule_reference.json").write_text(json.dumps(rule_metrics, indent=2), encoding="utf-8")
        print(
            f"All-anomaly baseline: F1={all_anomaly_metrics['F1']:.4f}, "
            f"FAR={all_anomaly_metrics['FAR']:.4f}, MAR={all_anomaly_metrics['MAR']:.4f}"
        )
        if include_zscore_reference:
            print(
                f"Z-score rule reference: F1={rule_metrics['F1']:.4f}, "
                f"FAR={rule_metrics['FAR']:.4f}, MAR={rule_metrics['MAR']:.4f}"
            )

        kb_info = initialize_knowledge_base(config, rebuild=True)
        retriever = RAGRetriever(config, kb_info)
        if args.skip_llm:
            comparison = {"All-anomaly baseline": all_anomaly_metrics}
            if include_zscore_reference:
                comparison["Z-score rule reference"] = rule_metrics
        else:
            without_detector = GenericLLMDetector(config, EXT / "prompts" / "hai_without_rag_prompt.txt")
            rag_detector = GenericLLMDetector(config, EXT / "prompts" / "hai_prompt.txt")
            without = run_llm_group(
                "LLM without RAG",
                windows,
                without_detector,
                retriever,
                False,
                results_dir / "without_rag",
                constrained_final=False,
            )
            with_rag = run_llm_group(
                "LLM with HAI-RAG",
                windows,
                rag_detector,
                retriever,
                True,
                results_dir / "with_rag",
                constrained_final=True,
            )
            comparison = {
                "All-anomaly baseline": all_anomaly_metrics,
                "LLM without RAG": without,
                "LLM with HAI-RAG": with_rag,
            }
            if include_zscore_reference:
                comparison = {
                    "All-anomaly baseline": all_anomaly_metrics,
                    "Z-score rule reference": rule_metrics,
                    "LLM without RAG": without,
                    "LLM with HAI-RAG": with_rag,
                }
        (results_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
