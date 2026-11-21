"""
数据处理模块

负责加载SKAB数据集、预处理、基线计算和滑动窗口切分。
"""

import os
import glob
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def scan_data_files(skab_dir: str, subdirs: List[str] = None) -> List[str]:
    """
    扫描数据目录下的所有CSV文件

    Args:
        skab_dir: SKAB数据集根目录
        subdirs: 子目录列表，默认为valve1, valve2, other

    Returns:
        CSV文件路径列表
    """
    if subdirs is None:
        subdirs = ['valve1', 'valve2', 'other']

    csv_files = []
    for subdir in subdirs:
        pattern = os.path.join(skab_dir, subdir, "*.csv")
        csv_files.extend(glob.glob(pattern))

    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    return csv_files


def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载单个CSV文件

    Args:
        file_path: CSV文件路径

    Returns:
        DataFrame或None
    """
    try:
        # CSV使用分号作为分隔符
        df = pd.read_csv(file_path, sep=';', parse_dates=['datetime'])
        return df
    except Exception as e:
        logger.warning(f"加载文件失败 {file_path}: {e}")
        return None


def get_column_names(df: pd.DataFrame) -> List[str]:
    """
    获取DataFrame的所有列名

    Args:
        df: 数据DataFrame

    Returns:
        列名列表
    """
    return list(df.columns)


def calculate_duration_minutes(df: pd.DataFrame) -> float:
    """
    计算数据时长（分钟）

    Args:
        df: 包含datetime列的DataFrame

    Returns:
        时长（分钟）
    """
    if len(df) < 2:
        return 0.0
    duration = df['datetime'].max() - df['datetime'].min()
    return duration.total_seconds() / 60.0


def filter_duration(df: pd.DataFrame, min_duration: float, discard_minutes: float) -> pd.DataFrame:
    """
    过滤时长并丢弃前N分钟数据

    Args:
        df: 原始DataFrame
        min_duration: 最小保留时长（分钟）
        discard_minutes: 丢弃开头分钟数

    Returns:
        处理后的DataFrame
    """
    # 丢弃前N分钟数据
    if discard_minutes > 0:
        start_time = df['datetime'].min()
        cutoff_time = start_time + pd.Timedelta(minutes=discard_minutes)
        df = df[df['datetime'] >= cutoff_time].reset_index(drop=True)

    # 检查剩余时长
    duration = calculate_duration_minutes(df)
    if duration < min_duration:
        return None

    return df


def compute_baseline_from_anomaly_free(anomaly_free_path: str, features: List[str]) -> Dict[str, Dict[str, float]]:
    """
    从anomaly-free.csv计算全局基线（均值和标准差）

    Args:
        anomaly_free_path: anomaly-free.csv文件路径
        features: 特征列表

    Returns:
        {特征名: {'mean': 均值, 'std': 标准差}}
    """
    df = load_csv_data(anomaly_free_path)
    if df is None:
        raise ValueError(f"无法加载anomaly-free文件: {anomaly_free_path}")

    baseline = {}
    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"特征 {feat} 不存在于数据中，可用列: {list(df.columns)}")
        baseline[feat] = {
            'mean': df[feat].mean(),
            'std': df[feat].std()
        }
        logger.info(f"基线 {feat}: mean={baseline[feat]['mean']:.4f}, std={baseline[feat]['std']:.4f}")

    return baseline


def compute_baseline_per_file(df: pd.DataFrame, features: List[str], baseline_points: int = 400) -> Dict[str, Dict[str, float]]:
    """
    从每个文件的前N点计算基线

    Args:
        df: DataFrame
        features: 特征列表
        baseline_points: 基线点数

    Returns:
        {特征名: {'mean': 均值, 'std': 标准差}}
    """
    baseline_df = df.head(baseline_points)

    baseline = {}
    for feat in features:
        baseline[feat] = {
            'mean': baseline_df[feat].mean(),
            'std': baseline_df[feat].std()
        }

    return baseline


def create_sliding_windows(df: pd.DataFrame, features: List[str],
                           window_size: int, window_step: int,
                           baseline: Dict[str, Dict[str, float]]) -> List[Dict]:
    """
    创建滑动窗口并计算统计量，包含时序差分特征

    Args:
        df: 处理后的DataFrame
        features: 特征列表
        window_size: 窗口大小
        window_step: 窗口步长
        baseline: 基线字典

    Returns:
        窗口列表，每个元素包含stats, label, file_name, window_idx
    """
    windows = []
    n_samples = len(df)

    # 检查是否有anomaly列
    has_anomaly = 'anomaly' in df.columns

    for start_idx in range(0, n_samples - window_size + 1, window_step):
        end_idx = start_idx + window_size
        window_df = df.iloc[start_idx:end_idx]

        # 计算统计量
        stats = {}
        for feat in features:
            values = window_df[feat]
            mean_val = values.mean()
            std_val = values.std()
            max_val = values.max()

            # 计算z_score
            if baseline[feat]['std'] > 0:
                z_score = (mean_val - baseline[feat]['mean']) / baseline[feat]['std']
            else:
                z_score = 0.0

            stats[feat] = {
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'z_score': z_score
            }

        # 计算跨特征的组合特征
        # 振动比率：两个加速度的比值
        if 'Accelerometer1RMS' in stats and 'Accelerometer2RMS' in stats:
            accel1_mean = stats['Accelerometer1RMS']['mean']
            accel2_mean = stats['Accelerometer2RMS']['mean']
            if abs(accel2_mean) > 1e-6:
                stats['Accel_Ratio'] = accel1_mean / accel2_mean
            else:
                stats['Accel_Ratio'] = 0.0

        # Rate-Temp组合：检测阀门问题模式
        if 'Volume Flow RateRMS' in stats and 'Temperature' in stats:
            rate_mean = stats['Volume Flow RateRMS']['mean']
            temp_mean = stats['Temperature']['mean']
            # 归一化后相乘
            rate_z = stats['Volume Flow RateRMS']['z_score']
            temp_z = stats['Temperature']['z_score']
            stats['Rate_Temp_Product'] = rate_z * temp_z

        # 窗口标签：窗口内任意点anomaly==1则标签为1
        label = 1 if has_anomaly and window_df['anomaly'].max() == 1 else 0

        windows.append({
            'stats': stats,
            'label': label,
            'file_name': '',
            'window_idx': len(windows)
        })

    return windows


def process_data_files(config: dict, sample: int = None) -> Tuple[List[Dict], Dict]:
    """
    处理所有数据文件

    Args:
        config: 配置字典
        sample: 只处理前N个文件

    Returns:
        (窗口列表, 基线字典)
    """
    skab_dir = config['data']['skab_dir']
    features = config['data']['features']
    baseline_mode = config['data']['baseline_mode']
    discard_minutes = config['data']['discard_minutes']
    min_duration_minutes = config['data']['min_duration_minutes']
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    baseline_points = config['data'].get('baseline_points', 400)  # 默认400

    # 扫描数据文件
    csv_files = scan_data_files(skab_dir)
    if sample:
        csv_files = csv_files[:sample]
        logger.info(f"只处理前 {sample} 个文件")

    # 加载anomaly-free计算基线（如果使用anomaly_free模式）
    baseline = None
    if baseline_mode == 'anomaly_free':
        anomaly_free_path = os.path.join(skab_dir, 'anomaly-free', 'anomaly-free.csv')
        baseline = compute_baseline_from_anomaly_free(anomaly_free_path, features)

    # 处理每个文件
    all_windows = []
    window_offset = 0

    for file_path in csv_files:
        logger.info(f"处理文件: {file_path}")
        df = load_csv_data(file_path)

        if df is None:
            continue

        # 打印列名确认数据结构
        logger.info(f"  列名: {get_column_names(df)}")

        # 检查是否有时长
        duration = calculate_duration_minutes(df)
        logger.info(f"  原始时长: {duration:.2f} 分钟")

        # 过滤时长并丢弃前N分钟
        df_filtered = filter_duration(df, min_duration_minutes, discard_minutes)
        if df_filtered is None:
            logger.info(f"  跳过: 时长不足")
            continue

        # 如果使用per_file模式，计算每个文件的基线
        if baseline_mode == 'per_file':
            baseline = compute_baseline_per_file(df_filtered, features, baseline_points)

        # 创建滑动窗口
        windows = create_sliding_windows(
            df_filtered, features, window_size, window_step, baseline
        )

        # 更新窗口信息
        file_name = os.path.basename(file_path)
        for w in windows:
            w['file_name'] = file_name
            w['window_idx'] += window_offset

        all_windows.extend(windows)
        window_offset = len(all_windows)

        logger.info(f"  生成 {len(windows)} 个窗口，累计 {len(all_windows)} 个")

    logger.info(f"总计生成 {len(all_windows)} 个窗口")
    return all_windows, baseline


def process_data_files_from_list(config: dict, csv_files: List[str]) -> Tuple[List[Dict], Dict]:
    """
    处理指定文件列表的数据

    Args:
        config: 配置字典
        csv_files: 要处理的CSV文件路径列表

    Returns:
        (窗口列表, 基线字典)
    """
    features = config['data']['features']
    baseline_mode = config['data']['baseline_mode']
    discard_minutes = config['data']['discard_minutes']
    min_duration_minutes = config['data']['min_duration_minutes']
    window_size = config['data']['window_size']
    window_step = config['data']['window_step']
    baseline_points = config['data'].get('baseline_points', 400)

    # 加载anomaly-free计算基线（如果使用anomaly_free模式）
    baseline = None
    if baseline_mode == 'anomaly_free':
        skab_dir = config['data']['skab_dir']
        anomaly_free_path = os.path.join(skab_dir, 'anomaly-free', 'anomaly-free.csv')
        baseline = compute_baseline_from_anomaly_free(anomaly_free_path, features)

    # 处理每个文件
    all_windows = []
    window_offset = 0

    for file_path in csv_files:
        logger.info(f"处理文件: {file_path}")
        df = load_csv_data(file_path)

        if df is None:
            continue

        logger.info(f"  列名: {get_column_names(df)}")

        duration = calculate_duration_minutes(df)
        logger.info(f"  原始时长: {duration:.2f} 分钟")

        df_filtered = filter_duration(df, min_duration_minutes, discard_minutes)
        if df_filtered is None:
            logger.info(f"  跳过: 时长不足")
            continue

        if baseline_mode == 'per_file':
            baseline = compute_baseline_per_file(df_filtered, features, baseline_points)

        windows = create_sliding_windows(
            df_filtered, features, window_size, window_step, baseline
        )

        file_name = os.path.basename(file_path)
        for w in windows:
            w['file_name'] = file_name
            w['window_idx'] += window_offset

        all_windows.extend(windows)
        window_offset = len(all_windows)

        logger.info(f"  生成 {len(windows)} 个窗口，累计 {len(all_windows)} 个")

    logger.info(f"总计生成 {len(all_windows)} 个窗口")
    return all_windows, baseline


if __name__ == "__main__":
    # 测试入口
    config = load_config()
    windows, baseline = process_data_files(config, sample=2)
    print(f"\n共处理 {len(windows)} 个窗口")
    if windows:
        print(f"第一个窗口统计: {windows[0]['stats']}")
        print(f"第一个窗口标签: {windows[0]['label']}")
