"""
RAG检索模块

负责检索与当前窗口z_score相关的知识，包括：
1. 查zscore_rules.csv获取特征偏差描述
2. FAISS索引检索Top-K相关知识条目
"""

import os
import logging
import pandas as pd
from typing import Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG检索器"""

    def __init__(self, config: dict, kb_info: dict):
        """
        初始化RAG检索器

        Args:
            config: 配置字典
            kb_info: 知识库信息字典
        """
        self.config = config
        self.kb_info = kb_info
        self.top_k = config['knowledge_base']['top_k']
        self.zscore_threshold = config['knowledge_base'].get('zscore_threshold', 1.5)

        # 加载zscore规则
        self.zscore_rules = self._load_zscore_rules()

        # 加载FAISS索引
        self.faiss_index = None
        self.texts = None
        self.embedding_model = None
        self._load_faiss_index()

    def _load_zscore_rules(self) -> pd.DataFrame:
        """加载zscore规则"""
        rules_path = self.kb_info['zscore_rules']
        df = pd.read_csv(rules_path, encoding='utf-8')
        # 转换 z_min 和 z_max 为浮点数
        df['z_min'] = df['z_min'].astype(float)
        df['z_max'] = df['z_max'].astype(float)
        logger.info(f"加载了 {len(df)} 条zscore规则")
        return df

    def _load_faiss_index(self) -> None:
        """加载FAISS索引"""
        index_path = self.kb_info['faiss_index']
        texts_path = self.kb_info['faiss_texts']

        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            logger.warning("FAISS索引不存在，跳过加载")
            return

        try:
            import faiss
            import pickle

            # faiss.read_index may fail on Windows paths containing non-ASCII
            # characters. Read with Python first, then deserialize in FAISS.
            with open(index_path, 'rb') as f:
                index_bytes = f.read()
            index_array = np.frombuffer(index_bytes, dtype='uint8')
            self.faiss_index = faiss.deserialize_index(index_array)

            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)

            logger.info(f"加载了FAISS索引，包含 {len(self.texts)} 个文本段落")
        except Exception as e:
            logger.warning(f"加载FAISS索引失败: {e}")

    def get_zscore_description(self, feature_name: str, z_score: float) -> str:
        """
        根据z_score获取偏差描述

        Args:
            feature_name: 特征名称
            z_score: z分数

        Returns:
            偏差描述字符串
        """
        # 查找对应的z_score区间
        for _, row in self.zscore_rules.iterrows():
            if row['z_min'] <= z_score < row['z_max']:
                return f"{feature_name}: z_score={z_score:.2f}, {row['level']}, {row['description']}"

        # 边界情况（使用-999和999作为-inf/inf的近似）
        if z_score >= 999:
            row = self.zscore_rules.iloc[-1]
            return f"{feature_name}: z_score={z_score:.2f}, {row['level']}, {row['description']}"
        if z_score < -999:
            row = self.zscore_rules.iloc[0]
            return f"{feature_name}: z_score={z_score:.2f}, {row['level']}, {row['description']}"

        return f"{feature_name}: z_score={z_score:.2f}, Unknown"

    def retrieve_faiss(self, query: str, top_k: int = None) -> List[str]:
        """
        从FAISS索引检索相关知识

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关知识文本列表
        """
        if self.faiss_index is None or self.texts is None:
            return []

        if top_k is None:
            top_k = self.top_k

        try:
            from sentence_transformers import SentenceTransformer

            # 加载模型（延迟加载）
            if self.embedding_model is None:
                model_name = self.config['knowledge_base']['embedding_model']
                self.embedding_model = SentenceTransformer(model_name)

            # 生成查询向量
            query_vec = self.embedding_model.encode([query])

            # 检索
            distances, indices = self.faiss_index.search(query_vec, top_k)

            # 返回相关文本
            results = []
            for idx in indices[0]:
                if idx < len(self.texts):
                    results.append(self.texts[idx])

            return results
        except Exception as e:
            logger.warning(f"FAISS检索失败: {e}")
            return []

    def retrieve(self, window_stats: Dict) -> str:
        """
        检索与当前窗口相关的所有知识

        Args:
            window_stats: 窗口统计字典 {特征名: {mean, std, max, z_score}}

        Returns:
            拼接好的RAG上下文字符串
        """
        context_parts = []

        # 1. 添加z_score描述（只处理原始特征，跳过组合特征）
        context_parts.append("=== Z-Score Analysis ===")
        for feature_name, stats in window_stats.items():
            # 跳过组合特征（它们是float而不是dict）
            if not isinstance(stats, dict):
                continue
            z_score = stats.get('z_score', 0)
            description = self.get_zscore_description(feature_name, z_score)
            context_parts.append(description)

        # 2. 构建检索查询，阈值用于控制进入RAG查询的显著偏移变量
        z_scores = {name: stats.get('z_score', 0) for name, stats in window_stats.items()
                    if isinstance(stats, dict)}
        abnormal_features = []
        for feature_name, z_score in z_scores.items():
            if abs(z_score) > self.zscore_threshold:
                direction = "high" if z_score > 0 else "low"
                abnormal_features.append(f"{feature_name}_{direction}")

        # 3. 始终检索领域知识（不仅异常时）
        context_parts.append("\n=== Domain Knowledge ===")
        if abnormal_features:
            query = " ".join(abnormal_features)
            relevant_knowledge = self.retrieve_faiss(query, self.top_k)
            for i, knowledge in enumerate(relevant_knowledge, 1):
                context_parts.append(f"[{i}] {knowledge}")
        else:
            # 无异常时检索正常行为模式
            normal_knowledge = self.retrieve_faiss("normal operation baseline", self.top_k)
            for i, knowledge in enumerate(normal_knowledge, 1):
                context_parts.append(f"[{i}] {knowledge}")

        return "\n".join(context_parts)


if __name__ == "__main__":
    # 测试入口
    import yaml
    from src.knowledge_base import initialize_knowledge_base

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化知识库
    kb_info = initialize_knowledge_base(config, rebuild=False)

    # 创建检索器
    retriever = RAGRetriever(config, kb_info)

    # 测试检索
    test_stats = {
        'Accelerometer1RMS': {'mean': 0.03, 'std': 0.001, 'max': 0.032, 'z_score': 3.5},
        'Accelerometer2RMS': {'mean': 0.04, 'std': 0.001, 'max': 0.042, 'z_score': 2.1},
        'Temperature': {'mean': 85.0, 'std': 1.0, 'max': 86.5, 'z_score': -0.5},
        'Volume Flow RateRMS': {'mean': 40.0, 'std': 2.0, 'max': 42.0, 'z_score': -2.8}
    }

    rag_context = retriever.retrieve(test_stats)
    print("=== RAG Context ===")
    print(rag_context)
