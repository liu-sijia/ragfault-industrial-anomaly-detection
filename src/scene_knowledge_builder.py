from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from knowledge_base import build_faiss_index, create_zscore_rules


DEFAULT_EXTRACT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_DOCUMENT_CHARS = 14000


def build_scene_knowledge_base(
    document_path: str | Path,
    csv_path: str | Path,
    output_root: str | Path,
    scene_name: str | None = None,
    extraction: dict[str, Any] | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EXTRACT_MODEL,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    build_index: bool = True,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    document_path = Path(document_path)
    csv_path = Path(csv_path)
    output_root = Path(output_root)

    document_text = read_document_text(document_path)
    csv_columns = read_csv_columns(csv_path)
    extraction_error = None
    extraction_mode = "provided"

    if extraction is None:
        try:
            extraction = extract_with_ollama(
                document_text=document_text,
                csv_columns=csv_columns,
                ollama_url=ollama_url,
                model=model,
                scene_name=scene_name,
            )
            extraction_mode = "ollama"
        except Exception as exc:
            if not allow_fallback:
                raise
            extraction_error = str(exc)
            extraction = fallback_extraction(document_text, csv_columns, scene_name or document_path.stem)
            extraction_mode = "fallback"

    normalized = normalize_extraction(extraction, csv_columns, scene_name)
    scene_id = make_scene_id(normalized["scene_name"])
    scene_dir = unique_scene_dir(output_root, scene_id)
    domain_dir = scene_dir / "domain_knowledge"
    domain_dir.mkdir(parents=True, exist_ok=True)

    create_zscore_rules(str(scene_dir))
    write_sensor_info(domain_dir / "sensor_info.txt", normalized)
    write_anomaly_patterns(domain_dir / "anomaly_patterns.txt", normalized)

    metadata = {
        "scene_id": scene_dir.name,
        "scene_name": normalized["scene_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_document": document_path.name,
        "source_csv": csv_path.name,
        "csv_columns": csv_columns,
        "matched_sensor_count": len([s for s in normalized["sensors"] if s.get("csv_field") in csv_columns]),
        "sensor_count": len(normalized["sensors"]),
        "anomaly_pattern_count": len(normalized["anomaly_patterns"]),
        "extract_model": model,
        "extraction_mode": extraction_mode,
        "extraction_error": extraction_error,
        "extraction": normalized,
    }
    (scene_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if build_index:
        old_index = scene_dir / "faiss_index"
        if old_index.exists():
            shutil.rmtree(old_index)
        build_faiss_index(str(scene_dir), embedding_model)

    return scene_result(scene_dir, metadata)


def read_document_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("当前环境未安装 python-docx，无法读取 DOCX 工程资料。") from exc
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("当前环境未安装 pypdf，无法读取 PDF 工程资料。") from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError("工程资料仅支持 txt、md、docx、pdf。")


def read_csv_columns(path: str | Path) -> list[str]:
    path = Path(path)
    errors: list[str] = []
    for kwargs in ({"sep": None, "engine": "python"}, {"sep": ";"}, {"sep": ","}):
        try:
            frame = pd.read_csv(path, nrows=5, **kwargs)
            columns = [str(column).strip() for column in frame.columns if str(column).strip()]
            if columns:
                return columns
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"无法读取 CSV 字段：{'; '.join(errors[:2])}")


def extract_with_ollama(
    document_text: str,
    csv_columns: list[str],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EXTRACT_MODEL,
    scene_name: str | None = None,
) -> dict[str, Any]:
    prompt = build_extraction_prompt(document_text, csv_columns, scene_name)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama 抽取服务不可用：{exc}") from exc

    raw = body.get("response", "")
    return parse_json_object(raw)


def build_extraction_prompt(document_text: str, csv_columns: list[str], scene_name: str | None) -> str:
    columns = ", ".join(csv_columns)
    clipped_doc = document_text[:MAX_DOCUMENT_CHARS]
    name_hint = scene_name or "请从资料中概括"
    return f"""
你是工业异常检测系统的知识库构建助手。请根据工程资料和 CSV 字段，抽取适合 RAG 检索的场景知识。

场景名称提示：{name_hint}
CSV 字段：{columns}

要求：
1. sensors 中的 csv_field 必须优先使用 CSV 字段中的原始列名；无法匹配时留空。
2. anomaly_patterns 要描述多变量关系，例如“流量降低且温度升高”。
3. 只返回 JSON，不要 Markdown，不要解释。

JSON 格式：
{{
  "scene_name": "场景名称",
  "sensors": [
    {{
      "name": "传感器名称",
      "csv_field": "CSV列名",
      "meaning": "测量含义",
      "unit": "单位或空字符串",
      "normal_behavior": "正常运行表现",
      "related_faults": ["相关故障"]
    }}
  ],
  "anomaly_patterns": [
    {{
      "name": "异常模式名称",
      "related_sensors": ["CSV列名或传感器名"],
      "condition": "变量组合条件",
      "explanation": "工程解释"
    }}
  ]
}}

工程资料：
{clipped_doc}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("LLM 未返回可解析的 JSON。")
    return json.loads(match.group(0))


def normalize_extraction(extraction: dict[str, Any], csv_columns: list[str], scene_name: str | None = None) -> dict[str, Any]:
    normalized_name = scene_name or str(extraction.get("scene_name") or "工业场景知识库").strip()
    sensors = []
    for item in extraction.get("sensors") or []:
        csv_field = best_column_match(str(item.get("csv_field") or item.get("name") or ""), csv_columns)
        sensors.append(
            {
                "name": str(item.get("name") or csv_field or "Unnamed Sensor").strip(),
                "csv_field": csv_field,
                "meaning": str(item.get("meaning") or "工业过程变量").strip(),
                "unit": str(item.get("unit") or "").strip(),
                "normal_behavior": str(item.get("normal_behavior") or "正常运行时保持在基线范围内。").strip(),
                "related_faults": listify(item.get("related_faults")),
            }
        )

    if not sensors:
        sensors = fallback_extraction("", csv_columns, normalized_name)["sensors"]

    patterns = []
    for item in extraction.get("anomaly_patterns") or []:
        patterns.append(
            {
                "name": str(item.get("name") or "异常模式").strip(),
                "related_sensors": listify(item.get("related_sensors")),
                "condition": str(item.get("condition") or "相关变量偏离基线范围。").strip(),
                "explanation": str(item.get("explanation") or "该模式可能对应设备状态变化或故障演化。").strip(),
            }
        )

    if not patterns:
        patterns = fallback_extraction("", csv_columns, normalized_name)["anomaly_patterns"]

    return {"scene_name": normalized_name, "sensors": sensors, "anomaly_patterns": patterns}


def fallback_extraction(document_text: str, csv_columns: list[str], scene_name: str) -> dict[str, Any]:
    feature_columns = [column for column in csv_columns if column.lower() not in {"datetime", "time", "timestamp", "anomaly", "label"}]
    sensors = [
        {
            "name": column,
            "csv_field": column,
            "meaning": f"{column} process variable",
            "unit": "",
            "normal_behavior": "正常运行时应围绕历史基线小幅波动。",
            "related_faults": ["process drift", "sensor fault", "equipment degradation"],
        }
        for column in feature_columns
    ]
    first = feature_columns[0] if feature_columns else "main variable"
    second = feature_columns[1] if len(feature_columns) > 1 else first
    patterns = [
        {
            "name": "单变量持续偏移",
            "related_sensors": [first],
            "condition": f"{first} 的 z-score 持续高于或低于正常范围",
            "explanation": "单个关键变量持续偏离基线时，可能对应传感器漂移、负载变化或局部设备退化。",
        },
        {
            "name": "多变量联合偏移",
            "related_sensors": feature_columns[:4],
            "condition": f"{first} 与 {second} 同时出现方向一致或相互耦合的偏移",
            "explanation": "多个变量同步偏移时，随机扰动的可能性降低，更可能反映工况切换或故障传播。",
        },
    ]
    guessed_name = scene_name or infer_scene_name(document_text) or "工业场景知识库"
    return {"scene_name": guessed_name, "sensors": sensors, "anomaly_patterns": patterns}


def write_sensor_info(path: Path, extraction: dict[str, Any]) -> None:
    lines = [f"{extraction['scene_name']} 传感器信息", ""]
    for index, sensor in enumerate(extraction["sensors"], 1):
        lines.extend(
            [
                f"{index}. {sensor['name']}",
                f"   - CSV字段: {sensor.get('csv_field') or '未匹配'}",
                f"   - 测量含义: {sensor['meaning']}",
                f"   - 单位: {sensor.get('unit') or '未提供'}",
                f"   - 正常表现: {sensor['normal_behavior']}",
                f"   - 相关故障: {', '.join(sensor.get('related_faults') or ['未提供'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_anomaly_patterns(path: Path, extraction: dict[str, Any]) -> None:
    lines = [f"{extraction['scene_name']} 典型异常模式", ""]
    for index, pattern in enumerate(extraction["anomaly_patterns"], 1):
        lines.extend(
            [
                f"{index}. {pattern['name']}",
                f"   - 相关变量: {', '.join(pattern.get('related_sensors') or ['未提供'])}",
                f"   - 判定线索: {pattern['condition']}",
                f"   - 工程解释: {pattern['explanation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Z-Score 检索提示",
            "- |z| < 1.0: 通常视为稳定运行。",
            "- 1.0 <= |z| < 2.0: 轻微偏移，需要结合相邻变量判断。",
            "- 2.0 <= |z| < 3.0: 明显偏移，优先检索相关异常模式。",
            "- |z| >= 3.0: 严重偏移，应重点检查故障解释和多变量组合。",
        ]
    )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def list_scene_knowledge_bases(default_kb_dir: str | Path, scenes_root: str | Path) -> list[dict[str, Any]]:
    scenes = [
        {
            "id": "default",
            "name": "默认SKAB知识库",
            "path": str(Path(default_kb_dir).resolve()),
            "builtin": True,
            "sensor_count": 4,
            "anomaly_pattern_count": 6,
        }
    ]
    scenes_root = Path(scenes_root)
    if not scenes_root.exists():
        return scenes

    for scene_dir in sorted(scenes_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        metadata_path = scene_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        scenes.append(scene_result(scene_dir, metadata))
    return scenes


def scene_result(scene_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    extraction = metadata.get("extraction") or {}
    sensor_preview = (scene_dir / "domain_knowledge" / "sensor_info.txt").read_text(encoding="utf-8")[:1200]
    pattern_preview = (scene_dir / "domain_knowledge" / "anomaly_patterns.txt").read_text(encoding="utf-8")[:1200]
    return {
        "id": metadata.get("scene_id") or scene_dir.name,
        "scene_id": metadata.get("scene_id") or scene_dir.name,
        "name": metadata.get("scene_name") or scene_dir.name,
        "path": str(scene_dir.resolve()),
        "builtin": False,
        "created_at": metadata.get("created_at"),
        "source_document": metadata.get("source_document"),
        "source_csv": metadata.get("source_csv"),
        "csv_columns": metadata.get("csv_columns") or [],
        "matched_sensor_count": metadata.get("matched_sensor_count", 0),
        "sensor_count": metadata.get("sensor_count", len(extraction.get("sensors") or [])),
        "anomaly_pattern_count": metadata.get("anomaly_pattern_count", len(extraction.get("anomaly_patterns") or [])),
        "extraction_mode": metadata.get("extraction_mode"),
        "extraction_error": metadata.get("extraction_error"),
        "sensor_preview": sensor_preview,
        "pattern_preview": pattern_preview,
    }


def make_scene_id(name: str) -> str:
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return ascii_name or f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def unique_scene_dir(output_root: Path, scene_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / scene_id
    if not candidate.exists():
        return candidate
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{scene_id}_{suffix}"


def best_column_match(value: str, columns: list[str]) -> str:
    if not value:
        return ""
    normalized = compact(value)
    for column in columns:
        if compact(column) == normalized:
            return column
    for column in columns:
        if normalized and (normalized in compact(column) or compact(column) in normalized):
            return column
    return value if value in columns else ""


def compact(value: str) -> str:
    return re.sub(r"[\s_\-()/]+", "", value).lower()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;，、]", value) if item.strip()]
    return [str(value).strip()]


def infer_scene_name(document_text: str) -> str | None:
    first_line = next((line.strip("# \t") for line in document_text.splitlines() if line.strip()), "")
    return first_line[:40] if first_line else None
