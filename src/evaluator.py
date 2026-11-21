"""
评估模块

负责计算异常检测评估指标，并对比三种方法：
1. Baseline（全部预测为1）
2. IsolationForest
3. LLM+RAG（本系统）
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        指标字典: Accuracy, Precision, Recall, F1, FAR, MAR
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True/False Positives/Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall (Same as True Positive Rate)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # False Alarm Rate (FAR) = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Missed Alarm Rate (MAR) = FN / (FN + TP)
    mar = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FAR': far,
        'MAR': mar
    }


def get_baseline_predictions(n_samples: int) -> List[int]:
    """
    获取Baseline预测（全1）

    Args:
        n_samples: 样本数

    Returns:
        预测列表（全1）
    """
    return [1] * n_samples


def train_isolation_forest(X_train: np.ndarray, contamination: float = 0.1) -> IsolationForest:
    """
    训练IsolationForest模型

    Args:
        X_train: 训练数据
        contamination: 污染比例

    Returns:
        训练好的模型
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def get_iforest_predictions(model: IsolationForest, X: np.ndarray) -> List[int]:
    """
    获取IsolationForest预测

    Args:
        model: 训练好的模型
        X: 数据

    Returns:
        预测列表（1=异常, 0=正常）
    """
    # IsolationForest返回-1为异常，1为正常
    predictions = model.predict(X)
    # 转换为1=异常，0=正常
    return [1 if p == -1 else 0 for p in predictions]


def extract_features(windows: List[Dict]) -> np.ndarray:
    """
    从窗口列表提取特征矩阵

    Args:
        windows: 窗口列表

    Returns:
        特征矩阵 (n_samples, n_features)
    """
    features = []
    for window in windows:
        stats = window['stats']
        # 提取z_score作为特征
        feature = [
            stats['Accelerometer1RMS']['z_score'],
            stats['Accelerometer2RMS']['z_score'],
            stats['Temperature']['z_score'],
            stats['Volume Flow RateRMS']['z_score']
        ]
        features.append(feature)

    return np.array(features)


def evaluate_all_methods(windows: List[Dict], llm_predictions: List[int],
                          baseline_data: Dict = None) -> Dict[str, Dict]:
    """
    评估所有方法

    Args:
        windows: 窗口列表
        llm_predictions: LLM预测列表
        baseline_data: 基线数据（用于训练IsolationForest）

    Returns:
        方法-指标字典
    """
    n_samples = len(windows)

    # 提取真实标签
    y_true = [w['label'] for w in windows]

    results = {}

    # 1. Baseline方法（全预测为1）
    baseline_preds = get_baseline_predictions(n_samples)
    baseline_metrics = calculate_metrics(y_true, baseline_preds)
    results['Baseline'] = baseline_metrics

    logger.info(f"Baseline: F1={baseline_metrics['F1']:.4f}, FAR={baseline_metrics['FAR']:.4f}, MAR={baseline_metrics['MAR']:.4f}")

    # 2. IsolationForest方法
    if baseline_data is not None:
        # 使用基线数据训练
        X_train = np.array([
            [baseline_data[feat]['mean'], baseline_data[feat]['std']]
            for feat in ['Accelerometer1RMS', 'Accelerometer2RMS', 'Temperature', 'Volume Flow RateRMS']
        ]).T

        # 简单处理：用随机数据训练或使用配置的比例
        # 实际使用时应该用正常的窗口数据训练
        # 这里简单使用所有数据的z_score
        X = extract_features(windows)
        # 用异常比例作为contamination
        contamination = sum(y_true) / len(y_true) if sum(y_true) > 0 else 0.1

        try:
            iforest = train_isolation_forest(X, contamination)
            if_preds = get_iforest_predictions(iforest, X)
            if_metrics = calculate_metrics(y_true, if_preds)
            results['IsolForest'] = if_metrics

            logger.info(f"IsolationForest: F1={if_metrics['F1']:.4f}, FAR={if_metrics['FAR']:.4f}, MAR={if_metrics['MAR']:.4f}")
        except Exception as e:
            logger.warning(f"IsolationForest训练失败: {e}")

    # 3. LLM+RAG方法
    llm_metrics = calculate_metrics(y_true, llm_predictions)
    results['Ours(LLM+RAG)'] = llm_metrics

    logger.info(f"LLM+RAG: F1={llm_metrics['F1']:.4f}, FAR={llm_metrics['FAR']:.4f}, MAR={llm_metrics['MAR']:.4f}")

    return results


def print_comparison_table(results: Dict[str, Dict]) -> str:
    """
    打印不同检测方法的指标对比表格

    Args:
        results: 方法-指标字典

    Returns:
        表格字符串
    """
    header = f"{'Method':<15} {'F1':>8} {'FAR%':>8} {'MAR%':>8}"
    separator = "-" * len(header)

    lines = [header, separator]

    # FAR和MAR转为百分比
    for method, metrics in results.items():
        far_pct = metrics['FAR'] * 100
        mar_pct = metrics['MAR'] * 100
        lines.append(f"{method:<15} {metrics['F1']:>8.2f} {far_pct:>8.1f} {mar_pct:>8.1f}")

    table = "\n".join(lines)
    return table


def save_results(results: Dict, output_dir: str) -> None:
    """
    保存评估结果

    Args:
        results: 评估结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"评估结果已保存: {metrics_path}")


if __name__ == "__main__":
    # 测试入口
    # 模拟数据
    windows = [
        {'stats': {'Accelerometer1RMS': {'z_score': 1.5}, 'Accelerometer2RMS': {'z_score': 0.5},
                   'Temperature': {'z_score': -0.2}, 'Volume Flow RateRMS': {'z_score': -2.5}}, 'label': 1},
        {'stats': {'Accelerometer1RMS': {'z_score': 0.1}, 'Accelerometer2RMS': {'z_score': 0.2},
                   'Temperature': {'z_score': 0.1}, 'Volume Flow RateRMS': {'z_score': 0.3}}, 'label': 0},
        {'stats': {'Accelerometer1RMS': {'z_score': 3.2}, 'Accelerometer2RMS': {'z_score': 2.8},
                   'Temperature': {'z_score': 1.5}, 'Volume Flow RateRMS': {'z_score': -3.1}}, 'label': 1},
    ]

    llm_preds = [1, 0, 1]

    results = evaluate_all_methods(windows, llm_preds)
    table = print_comparison_table(results)
    print("\n=== Comparison Table ===")
    print(table)
