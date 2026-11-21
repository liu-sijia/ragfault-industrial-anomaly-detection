from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def build_hai_subset(
    raw_gzip: str | Path,
    output_csv: str | Path,
    features: Iterable[str],
    source_start: int,
    source_end: int,
    segments: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a small HAI subset in a simple datetime/features/anomaly format."""
    raw_gzip = Path(raw_gzip)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = ["time", *features, "attack"]
    df = pd.read_csv(raw_gzip, compression="gzip", usecols=columns)
    if segments:
        parts = []
        for segment_id, segment in enumerate(segments):
            attack_start = int(segment["attack_start"])
            attack_end = int(segment["attack_end"])
            pre_margin = int(segment.get("pre_margin", 180))
            post_margin = int(segment.get("post_margin", 180))
            start = max(0, attack_start - pre_margin)
            end = min(len(df), attack_end + post_margin)
            part = df.iloc[start:end].copy()
            part["segment_id"] = segment_id
            part["source_index"] = range(start, end)
            parts.append(part)
        subset = pd.concat(parts, ignore_index=True)
    else:
        subset = df.iloc[source_start:source_end].copy().reset_index(drop=True)
        subset["segment_id"] = 0
        subset["source_index"] = range(source_start, source_end)
    subset = subset.rename(columns={"time": "datetime", "attack": "anomaly"})
    subset["datetime"] = pd.to_datetime(subset["datetime"])
    subset["anomaly"] = subset["anomaly"].astype(int)

    subset.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv
