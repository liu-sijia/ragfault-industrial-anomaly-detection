"""
主入口模块

整合所有模块，执行端到端异常检测流程：
1. 加载配置，初始化知识库
2. 加载并预处理数据
3. 逐窗口进行RAG检索和LLM推理
4. 评估并生成可视化
"""

import os
import sys
import json
import logging
import argparse
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import load_config, process_data_files
from knowledge_base import initialize_knowledge_base
from rag_retriever import RAGRetriever
from llm_detector import LLMDetector
from evaluator import evaluate_all_methods, print_comparison_table, save_results
from visualizer import generate_all_visualizations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_predictions(windows: List[dict], output_dir: str) -> None:
    """
    保存预测结果到CSV

    Args:
        windows: 窗口列表
        output_dir: 输出目录
    """
    records = []
    for w in windows:
        record = {
            'file_name': w.get('file_name', ''),
            'window_idx': w.get('window_idx', 0),
            'label': w.get('label', 0),
            'prediction': w.get('prediction', 0),
            'explanation': w.get('explanation', ''),
            'acc1_zscore': w['stats']['Accelerometer1RMS']['z_score'],
            'acc2_zscore': w['stats']['Accelerometer2RMS']['z_score'],
            'temp_zscore': w['stats']['Temperature']['z_score'],
            'rate_zscore': w['stats']['Volume Flow RateRMS']['z_score']
        }
        records.append(record)

    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"预测结果已保存: {output_path}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM+RAG异常检测系统')
    parser.add_argument('--config', type=str, default='config.example.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='SKAB数据路径')
    parser.add_argument('--mode', type=str, default=None, choices=['local', 'api'],
                        help='LLM模式: local或api')
    parser.add_argument('--model', type=str, default=None,
                        help='模型名')
    parser.add_argument('--window_size', type=int, default=None,
                        help='窗口大小')
    parser.add_argument('--rebuild_kb', action='store_true',
                        help='重建知识库索引')
    parser.add_argument('--sample', type=int, default=None,
                        help='只处理前N个文件')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.data_dir:
        config['data']['skab_dir'] = args.data_dir
    if args.mode:
        config['llm']['mode'] = args.mode
    if args.model:
        config['llm']['model'] = args.model
    if args.window_size:
        config['data']['window_size'] = args.window_size
        config['data']['window_step'] = args.window_size

    output_dir = config['output']['results_dir']

    # 步骤1: 初始化知识库
    logger.info("=== 步骤1: 初始化知识库 ===")
    kb_info = initialize_knowledge_base(config, rebuild=args.rebuild_kb)

    # 步骤2: 加载并处理数据
    logger.info("=== 步骤2: 加载并处理数据 ===")
    windows, baseline = process_data_files(config, sample=args.sample)

    if not windows:
        logger.error("没有处理任何窗口数据")
        return

    logger.info(f"共处理 {len(windows)} 个窗口")

    # 步骤3: 初始化RAG检索器和LLM检测器
    logger.info("=== 步骤3: 初始化检测器 ===")
    retriever = RAGRetriever(config, kb_info)
    detector = LLMDetector(config)

    # 步骤4: 多线程并行进行RAG检索和LLM推理
    logger.info("=== 步骤4: LLM推理 (多线程并行) ===")

    # 为每个窗口准备数据
    def process_window(window):
        """处理单个窗口"""
        # RAG检索
        rag_context = retriever.retrieve(window['stats'])
        # LLM推理
        prediction, explanation, success = detector.detect(window['stats'], rag_context)
        return window, prediction, explanation, success

    # 使用线程池并行处理
    max_workers = 4  # 并行线程数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_window, w): w for w in windows}

        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing windows"):
            window, prediction, explanation, success = future.result()
            window['prediction'] = prediction
            window['explanation'] = explanation
            window['success'] = success

    # 步骤5: 评估
    logger.info("=== 步骤5: 评估 ===")
    llm_predictions = [w['prediction'] for w in windows]

    # 提取基线数据用于IsolationForest
    baseline_data = None
    if config['data']['baseline_mode'] == 'anomaly_free':
        from data_processor import compute_baseline_from_anomaly_free
        anomaly_free_path = os.path.join(
            config['data']['skab_dir'],
            'anomaly-free',
            'anomaly-free.csv'
        )
        baseline_data = compute_baseline_from_anomaly_free(
            anomaly_free_path,
            config['data']['features']
        )

    results = evaluate_all_methods(windows, llm_predictions, baseline_data)

    # 打印对比表格
    table = print_comparison_table(results)
    print("\n=== Comparison Table (Table III) ===")
    print(table)

    # 保存结果
    logger.info("=== 步骤6: 保存结果 ===")
    os.makedirs(output_dir, exist_ok=True)
    save_predictions(windows, output_dir)
    save_results(results, output_dir)

    # 步骤6: 生成可视化
    logger.info("=== 步骤7: 生成可视化 ===")
    generate_all_visualizations(windows, results, output_dir)

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    main()
