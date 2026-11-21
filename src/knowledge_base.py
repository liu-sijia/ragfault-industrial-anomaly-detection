"""
知识库构建模块

负责生成并保存领域知识，包括zscore规则、传感器信息和异常模式，
并使用sentence-transformers和FAISS构建向量索引。
"""

import os
import logging
import pandas as pd
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_zscore_rules(kb_dir: str) -> str:
    """
    创建zscore规则CSV文件

    Args:
        kb_dir: 知识库根目录

    Returns:
        zscore_rules.csv文件路径
    """
    rules_path = os.path.join(kb_dir, "zscore_rules.csv")

    if os.path.exists(rules_path):
        logger.info(f"zscore规则已存在: {rules_path}")
        return rules_path

    # 7档z_score区间
    rules = [
        {"z_min": -float('inf'), "z_max": -3, "level": "严重偏低", "description": "数值严重低于正常范围，可能存在严重故障"},
        {"z_min": -3, "z_max": -2, "level": "偏低", "description": "数值低于正常范围，可能存在轻度异常"},
        {"z_min": -2, "z_max": -1, "level": "轻微偏低", "description": "数值略低于正常范围"},
        {"z_min": -1, "z_max": 1, "level": "正常", "description": "数值在正常范围内"},
        {"z_min": 1, "z_max": 2, "level": "轻微偏高", "description": "数值略高于正常范围"},
        {"z_min": 2, "z_max": 3, "level": "偏高", "description": "数值高于正常范围，可能存在轻度异常"},
        {"z_min": 3, "z_max": float('inf'), "level": "严重偏高", "description": "数值严重高于正常范围，可能存在严重故障"},
    ]

    df = pd.DataFrame(rules)
    os.makedirs(kb_dir, exist_ok=True)
    df.to_csv(rules_path, index=False, encoding='utf-8')
    logger.info(f"已创建zscore规则: {rules_path}")

    return rules_path


def create_sensor_info(kb_dir: str) -> str:
    """
    创建传感器信息文本文件

    Args:
        kb_dir: 知识库根目录

    Returns:
        sensor_info.txt文件路径
    """
    domain_dir = os.path.join(kb_dir, "domain_knowledge")
    sensor_path = os.path.join(domain_dir, "sensor_info.txt")

    if os.path.exists(sensor_path):
        logger.info(f"传感器信息已存在: {sensor_path}")
        return sensor_path

    content = """SKAB水循环系统传感器信息

1. Accelerometer1RMS (加速度传感器1RMS)
   - 位置: 轴承座1
   - 测量: 振动加速度有效值(RMS)
   - 正常范围: 0.02-0.05 g
   - 用途: 监测旋转设备(泵、电机)的机械振动

2. Accelerometer2RMS (加速度传感器2RMS)
   - 位置: 轴承座2
   - 测量: 振动加速度有效值(RMS)
   - 正常范围: 0.02-0.05 g
   - 用途: 监测另一端轴承的振动状态

3. Temperature (温度)
   - 位置: 管道出口处
   - 测量: 水温(摄氏度)
   - 正常范围: 75-95°C
   - 用途: 监测系统热平衡，检测过热或冷却异常

4. Volume Flow RateRMS (体积流量RMS)
   - 位置: 流量计
   - 测量: 流体体积流量有效值
   - 正常范围: 30-150 L/min
   - 用途: 监测水流速率，反映阀门开度和泵性能
"""

    os.makedirs(domain_dir, exist_ok=True)
    with open(sensor_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"已创建传感器信息: {sensor_path}")

    return sensor_path


def create_anomaly_patterns(kb_dir: str) -> str:
    """
    创建异常模式文本文件

    Args:
        kb_dir: 知识库根目录

    Returns:
        anomaly_patterns.txt文件路径
    """
    domain_dir = os.path.join(kb_dir, "domain_knowledge")
    patterns_path = os.path.join(domain_dir, "anomaly_patterns.txt")

    if os.path.exists(patterns_path):
        logger.info(f"异常模式已存在: {patterns_path}")
        return patterns_path

    content = """SKAB水循环系统典型故障模式

1. 阀门部分关闭 (Valve Partial Closure)
   - 主要特征: Volume Flow RateRMS显著下降
   - 次要特征: Temperature可能轻微上升(流速减慢导致加热)
   - 原因: 调节阀开度减小
   - 检测: RateRMS的z_score < -2

2. 轴承失衡 (Bearing Imbalance)
   - 主要特征: Accelerometer1RMS和/或Accelerometer2RMS显著上升
   - 次要特征: 振动频率异常
   - 原因: 轴承磨损、不平衡或损坏
   - 检测: 加速度传感器z_score > 2

3. 气穴现象 (Cavitation)
   - 主要特征: 多个传感器同时异常
   - 组合特征:
     * AccelerometerRMS上升(气泡破裂冲击)
     * Temperature可能升高(摩擦生热)
     * RateRMS波动增大
   - 原因: 入口压力过低导致液体汽化
   - 检测: 多个特征z_score同时偏离正常范围

4. 温度异常 (Temperature Anomaly)
   - 主要特征: Temperature显著偏离正常范围
   - 类型:
     * 温度过高: 冷却不足或泵过载(z_score > 2)
     * 温度过低: 冷却过度或流量过大(z_score < -2)
   - 原因: 冷却系统故障、阀门开度异常

5. 泵性能下降 (Pump Performance Degradation)
   - 主要特征: RateRMS持续偏低
   - 次要特征: 可能伴随振动增加
   - 原因: 泵叶轮磨损、密封老化

6. 传感器故障 (Sensor Fault)
   - 特征: 单个传感器读数异常(极端高或低)
   - 原因: 传感器漂移、线路故障
   - 检测: 单一特征z_score极端异常(>5或<-5)

## Combined Diagnostic Rules

- Valve closure: RateRMS z_score < -2 AND Temperature z_score > 1
- Bearing issue: (Accel1RMS OR Accel2RMS) z_score > 2.5
- Cavitation: Multiple features with |z_score| > 1.5 simultaneously
- Temperature anomaly: Temperature z_score > 2.5 OR < -2.5

## Z-Score Threshold Reference

- |z| < 1.0: Normal operation
- 1.0 < |z| < 2.0: Mild deviation - monitor closely
- 2.0 < |z| < 3.0: Moderate anomaly - likely fault
- |z| > 3.0: Severe anomaly - definite fault
"""

    os.makedirs(domain_dir, exist_ok=True)
    with open(patterns_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"已创建异常模式: {patterns_path}")

    return patterns_path


def build_faiss_index(kb_dir: str, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
    """
    使用sentence-transformers和FAISS构建向量索引

    Args:
        kb_dir: 知识库根目录
        embedding_model: embedding模型名称
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers或faiss-cpu未安装，跳过索引构建")
        return

    index_dir = os.path.join(kb_dir, "faiss_index")
    index_path = os.path.join(index_dir, "index.faiss")
    texts_path = os.path.join(index_dir, "texts.pkl")

    if os.path.exists(index_path) and os.path.exists(texts_path):
        logger.info(f"FAISS索引已存在: {index_path}")
        return

    # 加载文本
    sensor_path = os.path.join(kb_dir, "domain_knowledge", "sensor_info.txt")
    patterns_path = os.path.join(kb_dir, "domain_knowledge", "anomaly_patterns.txt")

    texts = []
    if os.path.exists(sensor_path):
        with open(sensor_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    if os.path.exists(patterns_path):
        with open(patterns_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    if not texts:
        logger.warning("没有找到领域知识文本，跳过索引构建")
        return

    # 分割文本为较小段落
    paragraphs = []
    for text in texts:
        # 按双换行分段（保持每个故障模式的完整性）
        sections = text.split('\n\n')
        for section in sections:
            section = section.strip()
            if section:
                paragraphs.append(section)

    logger.info(f"加载了 {len(paragraphs)} 个文本段落")

    # 生成embedding
    logger.info(f"加载embedding模型: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    logger.info("生成embeddings...")
    embeddings = model.encode(paragraphs, show_progress_bar=True)

    # 构建FAISS索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 保存。faiss.write_index 在 Windows 非 ASCII 路径下可能打不开文件，
    # 先序列化索引，再交给 Python 文件 API 写入，避免中文路径失败。
    os.makedirs(index_dir, exist_ok=True)
    serialized_index = faiss.serialize_index(index)
    with open(index_path, 'wb') as f:
        f.write(bytes(serialized_index))

    import pickle
    with open(texts_path, 'wb') as f:
        pickle.dump(paragraphs, f)

    logger.info(f"已创建FAISS索引: {index_path}")


def initialize_knowledge_base(config: dict, rebuild: bool = False) -> Dict:
    """
    初始化知识库

    Args:
        config: 配置字典
        rebuild: 是否重建索引

    Returns:
        包含各知识库文件路径的字典
    """
    kb_dir = config['knowledge_base']['dir']
    embedding_model = config['knowledge_base']['embedding_model']

    # 创建zscore规则
    zscore_rules_path = create_zscore_rules(kb_dir)

    # 创建领域知识
    create_sensor_info(kb_dir)
    create_anomaly_patterns(kb_dir)

    # 构建FAISS索引
    if rebuild:
        # 删除旧索引
        index_dir = os.path.join(kb_dir, "faiss_index")
        if os.path.exists(index_dir):
            import shutil
            shutil.rmtree(index_dir)
            logger.info("已删除旧索引")

    build_faiss_index(kb_dir, embedding_model)

    return {
        'kb_dir': kb_dir,
        'zscore_rules': zscore_rules_path,
        'sensor_info': os.path.join(kb_dir, "domain_knowledge", "sensor_info.txt"),
        'anomaly_patterns': os.path.join(kb_dir, "domain_knowledge", "anomaly_patterns.txt"),
        'faiss_index': os.path.join(kb_dir, "faiss_index", "index.faiss"),
        'faiss_texts': os.path.join(kb_dir, "faiss_index", "texts.pkl")
    }


if __name__ == "__main__":
    # 测试入口
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    kb_info = initialize_knowledge_base(config, rebuild=True)
    print(f"知识库初始化完成: {kb_info}")
