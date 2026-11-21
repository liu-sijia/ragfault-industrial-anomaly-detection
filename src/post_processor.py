"""
后处理模块

用于对LLM预测结果进行时序后处理，提升检测效果：
1. 孤立点检测 - 消除0→1→0或1→0→1的孤立误判
2. 多数投票 - 利用时序连续性平滑预测结果
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def apply_outlier_detection(predictions: List[int]) -> List[int]:
    """
    孤立点检测后处理

    模式 0→1→0 → 修正为 0→0→0
    模式 1→0→1 → 修正为 1→1→1

    Args:
        predictions: 原始预测列表

    Returns:
        修正后的预测列表
    """
    if len(predictions) < 3:
        return predictions.copy()

    result = predictions.copy()
    # 从1到n-2遍历，避免边界问题
    for i in range(1, len(result) - 1):
        if result[i-1] == result[i+1] and result[i] != result[i-1]:
            result[i] = result[i-1]  # 修正为与邻居一致

    return result


def apply_majority_voting(predictions: List[int], window: int = 2) -> List[int]:
    """
    多数投票后处理

    使用[i-window:i+window+1]范围内的预测进行多数投票

    Args:
        predictions: 原始预测列表
        window: 投票半径 (window=2 表示使用[i-2:i+3]共5个窗口)

    Returns:
        投票后的预测列表
    """
    n = len(predictions)
    result = predictions.copy()

    for i in range(n):
        # 确定投票范围，避免边界越界
        start = max(0, i - window)
        end = min(n, i + window + 1)

        # 收集投票范围内的预测
        votes = predictions[start:end]
        majority = Counter(votes).most_common(1)[0][0]
        result[i] = majority

    return result


def post_process_predictions(predictions: List[int],
                              methods: List[str] = ['outlier', 'voting'],
                              voting_window: int = 2) -> List[int]:
    """
    对预测结果应用后处理

    Args:
        predictions: 原始预测列表
        methods: 要应用的后处理方法列表，按顺序执行
                - 'outlier': 孤立点检测
                - 'voting': 多数投票
        voting_window: 多数投票的窗口半径

    Returns:
        后处理后的预测列表
    """
    result = predictions.copy()

    for method in methods:
        if method == 'outlier':
            result = apply_outlier_detection(result)
        elif method == 'voting':
            result = apply_majority_voting(result, voting_window)

    return result


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    计算评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        包含各项指标的字典
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float(np.sum(y_true == y_pred)) / len(y_true)
    prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0
    mar = float(fn) / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'F1': f1, 'FAR': far, 'MAR': mar,
        'Precision': prec, 'Recall': rec, 'Accuracy': acc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }


def evaluate_post_processing(windows: List[Dict],
                               original_metrics: Dict,
                               methods: List[str] = ['outlier', 'voting'],
                               voting_window: int = 2) -> Tuple[Dict, Dict]:
    """
    评估后处理效果

    Args:
        windows: 窗口列表，每个包含 'label' 和 'prediction' 键
        original_metrics: 后处理前的原始指标
        methods: 要应用的后处理方法
        voting_window: 多数投票窗口半径

    Returns:
        (后处理后的指标, 改善量)
    """
    # 提取原始预测
    y_true = [w['label'] for w in windows]
    y_pred = [w['prediction'] for w in windows]

    # 应用后处理
    y_pred_processed = post_process_predictions(y_pred, methods, voting_window)

    # 计算后处理后的指标
    processed_metrics = calculate_metrics(y_true, y_pred_processed)

    # 计算改善量
    improvement = {
        'F1_delta': processed_metrics['F1'] - original_metrics['F1'],
        'FAR_delta': processed_metrics['FAR'] - original_metrics['FAR'],
        'MAR_delta': processed_metrics['MAR'] - original_metrics['MAR'],
    }

    return processed_metrics, improvement


if __name__ == "__main__":
    # 简单测试
    print("=== 后处理模块测试 ===")

    # 测试孤立点检测
    test_preds = [0, 0, 1, 0, 0, 1, 1, 1, 0]
    print(f"原始预测:  {test_preds}")
    print(f"孤立点检测: {apply_outlier_detection(test_preds)}")

    # 测试多数投票
    test_preds2 = [0, 0, 1, 0, 0]
    print(f"原始预测:  {test_preds2}")
    print(f"多数投票:  {apply_majority_voting(test_preds2, window=1)}")

    # 模拟评估
    windows = [
        {'label': 0, 'prediction': 0},
        {'label': 0, 'prediction': 1},  # 孤立误报
        {'label': 0, 'prediction': 0},
        {'label': 1, 'prediction': 1},
        {'label': 1, 'prediction': 1},
    ]

    original = {'F1': 0.7, 'FAR': 0.2, 'MAR': 0.3}
    processed, improvement = evaluate_post_processing(windows, original)
    print(f"\n原始指标:  F1={original['F1']:.4f}, FAR={original['FAR']:.4f}, MAR={original['MAR']:.4f}")
    print(f"处理后:    F1={processed['F1']:.4f}, FAR={processed['FAR']:.4f}, MAR={processed['MAR']:.4f}")
    print(f"改善:      F1={improvement['F1_delta']:+.4f}, FAR={improvement['FAR_delta']:+.4f}, MAR={improvement['MAR_delta']:+.4f}")