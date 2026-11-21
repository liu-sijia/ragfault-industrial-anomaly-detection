"""
可视化模块

负责生成可视化图表：
1. 时序图：4个特征时序，标注真实异常区间和预测窗口
2. 柱状图：三个方法F1/FAR/MAR对比
3. 热力图：本系统混淆矩阵
"""

import os
import logging
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_time_series(windows: List[Dict], file_name: str, output_dir: str) -> None:
    """
    绘制时序图

    Args:
        windows: 窗口列表
        file_name: 文件名
        output_dir: 输出目录
    """
    # 筛选指定文件的窗口
    file_windows = [w for w in windows if w.get('file_name') == file_name]

    if not file_windows:
        logger.warning(f"未找到文件 {file_name} 的窗口")
        return

    # 提取特征数据
    features = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Temperature', 'Volume Flow RateRMS']
    feature_data = {feat: [] for feat in features}
    labels = []
    predictions = []

    for w in file_windows:
        for feat in features:
            feature_data[feat].append(w['stats'][feat]['z_score'])
        labels.append(w.get('label', 0))
        predictions.append(w.get('prediction', 0))

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Time Series Analysis: {file_name}', fontsize=14)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, feat in enumerate(features):
        ax = axes[i]
        z_scores = feature_data[feat]

        # 绘制z_score曲线
        ax.plot(z_scores, color=colors[i], linewidth=1, label=f'{feat} z-score')

        # 标注真实异常区间
        for j, label in enumerate(labels):
            if label == 1:
                ax.axvspan(j - 0.5, j + 0.5, alpha=0.3, color='red', label='_nolegend_')

        # 标注预测异常
        for j, pred in enumerate(predictions):
            if pred == 1:
                ax.axvline(j, alpha=0.5, color='green', linewidth=2)

        # 绘制z=±2参考线
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)

        ax.set_ylabel(feat, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Window Index')
    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'time_series.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"时序图已保存: {output_path}")


def plot_comparison_bar(results: Dict[str, Dict], output_dir: str) -> None:
    """
    绘制方法对比柱状图

    将F1（越高越好）和FAR/MAR（越低越好）分开展示，避免量纲和语义不一致

    Args:
        results: 评估结果字典
        output_dir: 输出目录
    """
    methods = list(results.keys())
    f1_scores = [results[m]['F1'] for m in methods]
    far_scores = [results[m]['FAR'] * 100 for m in methods]  # 转为百分比
    mar_scores = [results[m]['MAR'] * 100 for m in methods]

    x = np.arange(len(methods))

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: F1分数 (越高越好)
    bars1 = ax1.bar(x, f1_scores, color='#1f77b4', width=0.6)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score (Higher is Better)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, axis='y', alpha=0.3)

    # F1数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图2: FAR和MAR (越低越好)
    width = 0.35
    bars2 = ax2.bar(x - width/2, far_scores, width, label='FAR%', color='#ff7f0e')
    bars3 = ax2.bar(x + width/2, mar_scores, width, label='MAR%', color='#2ca02c')

    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('FAR & MAR (Lower is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3)

    # FAR/MAR数值标签
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"对比图已保存: {output_path}")


def plot_confusion_matrix(windows: List[Dict], output_dir: str) -> None:
    """
    绘制混淆矩阵热力图

    Args:
        windows: 窗口列表（包含label和prediction）
        output_dir: 输出目录
    """
    # 提取真实标签和预测
    y_true = [w['label'] for w in windows]
    y_pred = [w.get('prediction', 0) for w in windows]

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'Anomaly (1)'],
                yticklabels=['Normal (0)', 'Anomaly (1)'],
                ax=ax)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix: LLM+RAG', fontsize=14)

    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"混淆矩阵已保存: {output_path}")


def generate_all_visualizations(windows: List[Dict], results: Dict[str, Dict],
                                 output_dir: str) -> None:
    """
    生成所有可视化

    Args:
        windows: 窗口列表
        results: 评估结果
        output_dir: 输出目录
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 图1：时序图（选择第一个有异常的文件）
    file_counts = {}
    for w in windows:
        fname = w.get('file_name', '')
        if fname:
            file_counts[fname] = file_counts.get(fname, 0) + w.get('label', 0)

    # 选择异常最多的文件
    if file_counts:
        selected_file = max(file_counts, key=file_counts.get)
        if file_counts[selected_file] > 0:
            plot_time_series(windows, selected_file, figures_dir)

    # 图2：方法对比
    plot_comparison_bar(results, figures_dir)

    # 图3：混淆矩阵
    plot_confusion_matrix(windows, figures_dir)

    logger.info(f"所有可视化已保存到: {figures_dir}")


if __name__ == "__main__":
    # 测试入口
    windows = [
        {'stats': {'Accelerometer1RMS': {'z_score': 1.5}, 'Accelerometer2RMS': {'z_score': 0.5},
                   'Temperature': {'z_score': -0.2}, 'Volume Flow RateRMS': {'z_score': -2.5}},
         'label': 1, 'prediction': 1, 'file_name': 'test.csv'},
        {'stats': {'Accelerometer1RMS': {'z_score': 0.1}, 'Accelerometer2RMS': {'z_score': 0.2},
                   'Temperature': {'z_score': 0.1}, 'Volume Flow RateRMS': {'z_score': 0.3}},
         'label': 0, 'prediction': 0, 'file_name': 'test.csv'},
        {'stats': {'Accelerometer1RMS': {'z_score': 3.2}, 'Accelerometer2RMS': {'z_score': 2.8},
                   'Temperature': {'z_score': 1.5}, 'Volume Flow RateRMS': {'z_score': -3.1}},
         'label': 1, 'prediction': 1, 'file_name': 'test.csv'},
    ]

    results = {
        'Baseline': {'F1': 0.62, 'FAR': 1.0, 'MAR': 0.0},
        'IsolForest': {'F1': 0.75, 'FAR': 0.2, 'MAR': 0.15},
        'Ours(LLM+RAG)': {'F1': 0.80, 'FAR': 0.1, 'MAR': 0.1}
    }

    generate_all_visualizations(windows, results, './results')
