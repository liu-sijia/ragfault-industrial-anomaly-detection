from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests


class GenericLLMDetector:
    """Prompt-based detector for extension experiments with arbitrary features."""

    def __init__(self, config: dict[str, Any], prompt_path: str | Path):
        self.config = config
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.model = config["llm"]["model"]
        self.local_url = config["llm"]["local_url"].rstrip("/")
        self.temperature = config["llm"].get("temperature", 0.0)
        self.max_tokens = config["llm"].get("max_tokens", 180)
        self.timeout = config["llm"].get("timeout", 500)
        self.severe_z_threshold = config["experiment"].get("severe_z_threshold", 2.5)
        self.multi_z_threshold = config["experiment"].get("multi_z_threshold", 1.5)

    def build_feature_table(self, window_stats: dict[str, Any]) -> str:
        lines = ["Variable | z_score | deviation_level", "--- | ---: | ---"]
        for feature, stats in window_stats.items():
            if not isinstance(stats, dict):
                continue
            z_score = float(stats.get("z_score", 0.0))
            lines.append(f"{feature} | {z_score:.2f} | {self._level(z_score)}")
        return "\n".join(lines)

    def build_prompt(self, window_stats: dict[str, Any], rag_context: str) -> str:
        return self.prompt_template.format(
            feature_zscore_table=self.build_feature_table(window_stats),
            rag_context=rag_context,
            severe_z_threshold=self.severe_z_threshold,
            multi_z_threshold=self.multi_z_threshold,
        )

    def detect(self, window_stats: dict[str, Any], rag_context: str) -> tuple[int, str, str]:
        prompt = self.build_prompt(window_stats, rag_context)
        response = self._call_ollama(prompt)
        prediction, reason = self.parse_response(response)
        return prediction, reason, prompt

    def parse_response(self, response: str) -> tuple[int, str]:
        if not response:
            return 0, "No response"

        decision_match = re.search(r"Decision:\s*(YES|NO)", response, re.I)
        if decision_match:
            prediction = 1 if decision_match.group(1).upper() == "YES" else 0
        elif re.search(r"\bYES\b", response, re.I):
            prediction = 1
        else:
            prediction = 0

        reason_match = re.search(r"Reason:\s*(.+)", response, re.I | re.S)
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = " ".join(line.strip() for line in response.splitlines() if line.strip())
        return prediction, reason[:350]

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        response = requests.post(f"{self.local_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    @staticmethod
    def _level(z_score: float) -> str:
        abs_z = abs(z_score)
        direction = "High" if z_score > 0 else "Low"
        if abs_z < 1.0:
            return "Normal"
        if abs_z < 1.5:
            return f"Slightly {direction}"
        if abs_z < 2.0:
            return f"Moderately {direction}"
        if abs_z < 3.0:
            return f"Clearly {direction}"
        return f"Severely {direction}"
