"""
LLM检测器模块

负责使用LLM进行异常检测推理，支持两种模式：
1. local: Ollama本地模型
2. api: OpenAI兼容API
"""

import os
import re
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are an expert anomaly detection system for industrial water circulation.

## DATA (Z-Scores - deviation from normal baseline)
Accel1 z-score: {acc1_zscore} | Accel2 z-score: {acc2_zscore} | Temp z-score: {temp_zscore} | Rate z-score: {rate_zscore}

## RAG CONTEXT
{rag_context}

## DECISION RULES

### Say YES (Anomaly) if ANY of these is TRUE:
1. At least 2 features with |z| > {zscore_threshold} simultaneously
2. Accel1 z > 2.0 OR Accel2 z > 2.0 (high vibration)
3. Rate z < -2.0 AND Temp z > 1.0 (valve problem pattern)
4. Temperature z < -3.5 OR Temperature z > 3.5 (extreme temperature)
5. Accel Ratio > 3.0 OR < 0.3 (bearing fault indicator)

### Say NO (Normal) if ALL of these are TRUE:
1. All |z| < {zscore_threshold}
2. Accel Ratio between 0.5 and 2.0 (normal vibration balance)

## OUTPUT (Format exactly):
Decision: YES/NO
Reason: brief explanation
"""


class LLMDetector:
    """LLM异常检测器"""

    def __init__(self, config: dict):
        """
        初始化LLM检测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.mode = config['llm']['mode']
        self.model = config['llm']['model']
        self.timeout = config['llm'].get('timeout', 120)
        self.max_tokens = config['llm'].get('max_tokens', 300)
        self.temperature = config['llm'].get('temperature', 0.1)
        self.request_interval = config['llm'].get('request_interval', 0)
        self._last_request_time = 0

    def build_prompt(self, window_stats: Dict, rag_context: str) -> str:
        """
        构建Prompt

        Args:
            window_stats: 窗口统计字典
            rag_context: RAG上下文

        Returns:
            完整Prompt字符串
        """
        # 提取z_score
        acc1_stats = window_stats.get('Accelerometer1RMS', {})
        acc2_stats = window_stats.get('Accelerometer2RMS', {})
        temp_stats = window_stats.get('Temperature', {})
        rate_stats = window_stats.get('Volume Flow RateRMS', {})
        zscore_threshold = self.config.get('knowledge_base', {}).get('zscore_threshold', 1.5)

        acc1_zscore = acc1_stats.get('z_score', 0)
        acc2_zscore = acc2_stats.get('z_score', 0)
        temp_zscore = temp_stats.get('z_score', 0)
        rate_zscore = rate_stats.get('z_score', 0)

        return PROMPT_TEMPLATE.format(
            acc1_zscore=f"{acc1_zscore:.2f}",
            acc2_zscore=f"{acc2_zscore:.2f}",
            temp_zscore=f"{temp_zscore:.2f}",
            rate_zscore=f"{rate_zscore:.2f}",
            rag_context=rag_context,
            zscore_threshold=f"{zscore_threshold:.1f}"
        )

    def call_local(self, prompt: str) -> Optional[str]:
        """
        调用Ollama本地模型

        Args:
            prompt: Prompt字符串

        Returns:
            模型响应或None
        """
        import requests

        url = self.config['llm']['local_url'] + "/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama请求超时: {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"无法连接到Ollama: {self.config['llm']['local_url']}")
            return None
        except Exception as e:
            logger.warning(f"Ollama调用失败: {e}")
            return None

    def call_llm(self, prompt: str) -> Tuple[Optional[str], int]:
        """
        调用本地Ollama

        Args:
            prompt: Prompt字符串

        Returns:
            (模型响应, 重试次数)
        """
        result = self.call_local(prompt)
        if result:
            return result, 0
        return None, -1

    def parse_response(self, response: str) -> Tuple[int, str]:
        """
        解析LLM响应

        Args:
            response: LLM响应字符串

        Returns:
            (预测标签, 解释)
            预测标签: 0=正常, 1=异常
            解释: 异常解释
        """
        if not response:
            return 0, "No response"

        prediction = 0
        explanation = ""

        # 新格式: Decision: YES/NO
        decision_match = re.search(r'Decision:\s*(YES|NO)', response, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
            prediction = 1 if decision == 'YES' else 0
        else:
            # 回退: 查找包含YES的行
            if re.search(r'\bYES\b', response, re.IGNORECASE):
                for line in response.split('\n'):
                    if 'Decision' in line and 'YES' in line:
                        prediction = 1
                        break

        # 提取Reason
        reason_match = re.search(r'Reason:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            explanation = reason_match.group(1).strip()[:200]
        else:
            # 提取最后一行为解释
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            if lines:
                explanation = lines[-1][:200]

        return prediction, explanation

    def detect(self, window_stats: Dict, rag_context: str) -> Tuple[int, str, bool]:
        """
        执行异常检测（纯LLM+RAG推理，无后处理覆盖）

        Args:
            window_stats: 窗口统计字典
            rag_context: RAG上下文

        Returns:
            (预测标签, 解释, 是否成功)
            预测标签: 0=正常, 1=异常
            解释: 异常解释
            是否成功: False表示API失败，需要特殊处理
        """
        prompt = self.build_prompt(window_stats, rag_context)
        response, retries = self.call_llm(prompt)

        if response is None:
            # API失败，标记为失败由调用者处理
            logger.error("LLM调用失败，标记为待处理样本")
            return 0, "API_FAILED", False

        prediction, explanation = self.parse_response(response)

        # 完全依靠LLM+RAG判断，无后处理覆盖
        # 这样保证算法的泛化能力和可迁移性
        return prediction, explanation, True


if __name__ == "__main__":
    # 测试入口
    import yaml

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    detector = LLMDetector(config)

    # 测试构建prompt
    test_stats = {
        'Accelerometer1RMS': {'mean': 0.03, 'std': 0.001, 'max': 0.032, 'z_score': 3.5},
        'Accelerometer2RMS': {'mean': 0.04, 'std': 0.001, 'max': 0.042, 'z_score': 2.1},
        'Temperature': {'mean': 85.0, 'std': 1.0, 'max': 86.5, 'z_score': -0.5},
        'Volume Flow RateRMS': {'mean': 40.0, 'std': 2.0, 'max': 42.0, 'z_score': -2.8}
    }

    rag_context = "Test RAG context"

    prompt = detector.build_prompt(test_stats, rag_context)
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Response Parsing Test ===")
    test_response = """* Is high deviation present for Accelerometer1RMS? Yes
* Is high deviation present for Accelerometer2RMS? Yes
* Is high deviation present for Temperature? No
* Is high deviation present for Volume Flow RateRMS? Yes
* Overall: Is this time window anomalous? Yes
* One-sentence explanation: The high vibration and low flow rate indicate a potential pump issue."""

    pred, exp = detector.parse_response(test_response)
    print(f"Prediction: {pred}")
    print(f"Explanation: {exp}")
