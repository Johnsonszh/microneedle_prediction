# 基础库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn
import torch.backends.mkldnn  # 替换原来的 torch.utils.mkl

# 系统和工具库
import json
import sys
import os
import platform
import warnings
import logging
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple
from enum import Enum

# 禁用警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('microneedle_quality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'platform': 'platform',
        'multiprocessing': 'multiprocessing'
    }

    missing_packages = []
    version_info = {}

    for package, import_name in required_packages.items():
        try:
            module = __import__(import_name)
            if hasattr(module, '__version__'):
                version_info[package] = module.__version__
            else:
                version_info[package] = "版本未知"
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("错误: 以下必要的包未安装:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请使用pip安装这些包:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        logger.info("所有依赖检查通过")
        logger.info("包版本信息:")
        for package, version in version_info.items():
            logger.info(f"  {package}: {version}")


def check_cuda_availability():
    """检查CUDA是否可用并打印设备信息"""
    device = None
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\nCUDA可用！使用设备: {torch.cuda.get_device_name(0)}")

            # 显示显存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            allocated_memory = torch.cuda.memory_allocated(0) / 1024 ** 3
            cached_memory = torch.cuda.memory_reserved(0) / 1024 ** 3

            print(f"总显存: {total_memory:.1f} GB")
            print(f"已分配显存: {allocated_memory:.1f} GB")
            print(f"缓存显存: {cached_memory:.1f} GB")

            # 检查CUDA版本
            cuda_version = torch.version.cuda
            print(f"CUDA版本: {cuda_version}")

            # 检查cuDNN版本
            if hasattr(torch.backends, 'cudnn'):
                cudnn_version = torch.backends.cudnn.version()
                cudnn_enabled = torch.backends.cudnn.enabled
                print(f"cuDNN版本: {cudnn_version}")
                print(f"cuDNN启用状态: {'是' if cudnn_enabled else '否'}")

            # 设置GPU性能优化
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("已启用CUDA性能优化")
        else:
            device = torch.device("cpu")
            print("\nCUDA不可用，使用CPU进行计算")
            print("警告：使用CPU训练可能会显著降低运行速度")

            # 显示CPU信息
            print(f"CPU型号: {platform.processor()}")
            print(f"CPU核心数: {multiprocessing.cpu_count()}")

            # 检查CPU优化状态
            if torch.backends.mkldnn.is_available():
                print("MKL-DNN加速: 可用")
                torch.backends.mkldnn.enabled = True
            else:
                print("MKL-DNN加速: 不可用")

            # 设置线程数
            torch.set_num_threads(multiprocessing.cpu_count())
            logger.info(f"已设置CPU线程数: {multiprocessing.cpu_count()}")

    except Exception as e:
        logger.error(f"检查设备状态时出错: {str(e)}")
        device = torch.device("cpu")
        print("\n检查设备状态时出错，将使用CPU进行计算")

    # 记录设备信息
    logger.info(f"使用计算设备: {device}")
    return device


# 在程序开始时运行依赖检查
check_dependencies()


# 定义打印方式枚举
class PrintMode(Enum):
    CONTINUOUS = 0  # 连续打印
    INTERMITTENT = 1  # 间歇打印
    RECIPROCATING = 2  # 往复打印


# 定义输入特征和质量评分组件
input_features = [
    'layer_height',  # 打印层高
    'exposure_time',  # 曝光时间
    'exposure_intensity',  # 曝光强度
    'bottom_layers',  # 底层层数
    'bottom_exposure_intensity',  # 底曝亮度
    'lifting_speed',  # 抬升速度
    'exposure_wait',  # 曝光前等待
    'liquid_speed',  # 入液速度
    'bottom_exposure_time',  # 底部曝光时间
    'print_mode'  # 打印方式
]

quality_score_components = [
    'needle_definition',  # 针尖清晰度
    'layer_adhesion',  # 层间粘结性
    'needle_height',  # 针尖高度
    'base_thickness',  # 底座厚度
    'material_curing'  # 材料固化度
]

quality_score_weights = {
    'needle_definition': 0.3,
    'layer_adhesion': 0.2,
    'needle_height': 0.2,
    'base_thickness': 0.15,
    'material_curing': 0.15
}

# 定义参数的合理范围
PARAM_RANGES = {
    'layer_height': (0.01, 0.1, 0.01),  # 范围：0.01-0.1mm，步长0.01
    'exposure_time': (1.0, 10.0, 0.5),  # 范围：1-10秒，步长0.5
    'exposure_intensity': (50, 100, 5),  # 范围：50-100%，步长5
    'bottom_layers': (3, 8, 1),  # 范围：3-8层，步长1
    'bottom_exposure_intensity': (60, 100, 5),  # 范围：60-100%，步长5
    'lifting_speed': (0.5, 3.0, 0.5),  # 范围：0.5-3.0mm/s，步长0.5
    'exposure_wait': (0.1, 2.0, 0.1),  # 范围：0.1-2.0秒，步长0.1
    'liquid_speed': (0.5, 3.0, 0.5),  # 范围：0.5-3.0mm/s，步长0.5
    'bottom_exposure_time': (10, 30, 2),  # 范围：10-30秒，步长2
    'print_mode': (0, 2, 1)  # 打印方式：0-连续，1-间歇，2-往复
}


def safe_input(prompt, default=None):
    """安全处理用户输入，包含中断处理"""
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print("\n输入被中断。" + ("使用默认值。" if default else "退出程序。"))
        if default is not None:
            return default
        sys.exit(0)


def get_parameter_unit(param: str) -> str:
    """获取参数的单位"""
    units = {
        'layer_height': 'mm',
        'exposure_time': '秒',
        'exposure_intensity': '%',
        'bottom_layers': '层',
        'bottom_exposure_intensity': '%',
        'lifting_speed': 'mm/s',
        'exposure_wait': '秒',
        'liquid_speed': 'mm/s',
        'bottom_exposure_time': '秒',
        'print_mode': ''
    }
    return units.get(param, '')


def get_print_mode_name(mode_value: int) -> str:
    """获取打印方式的名称"""
    mode_names = {
        0: "连续打印",
        1: "间歇打印",
        2: "往复打印"
    }
    return mode_names.get(mode_value, "未知")


def input_params():
    """获取用户输入的参数值"""
    params = {}
    print("\n请输入以下参数：")
    print("\n参数说明：")
    print("- layer_height: 打印层高，影响打印精度和时间")
    print("- exposure_time: 光固化时间，影响材料固化程度")
    print("- exposure_intensity: 光照强度，影响固化效果")
    print("- bottom_layers: 底层数量，影响基底牢固程度")
    print("- bottom_exposure_intensity: 底层曝光强度")
    print("- lifting_speed: 构建平台抬升速度")
    print("- exposure_wait: 曝光前等待时间，用于材料稳定")
    print("- liquid_speed: 料槽进料速度")
    print("- bottom_exposure_time: 底层曝光时间")
    print("- print_mode: 打印模式(0-连续，1-间歇，2-往复)")

    try:
        for feature in input_features:
            while True:
                try:
                    if feature == 'print_mode':
                        print("\n打印方式选项：")
                        print("0 - 连续打印: 打印速度快，适合简单结构")
                        print("1 - 间歇打印: 层间结合好，打印时间长")
                        print("2 - 往复打印: 可能影响表面质量，需要调整速度")
                        value_str = safe_input(f"{feature} [0-2]: ")
                    else:
                        min_val, max_val, step = PARAM_RANGES[feature]
                        unit = get_parameter_unit(feature)
                        value_str = safe_input(f"{feature} [{min_val}-{max_val}{unit}]: ")

                    value = float(value_str)

                    # 验证参数范围
                    min_val, max_val, _ = PARAM_RANGES[feature]
                    if not min_val <= value <= max_val:
                        raise ValueError(f"输入值超出范围 ({min_val}-{max_val})")

                    params[feature] = value
                    break
                except ValueError as e:
                    print(f"错误: {str(e)}")
                    print("请重新输入\n")

        return params

    except KeyboardInterrupt:
        logger.info("参数输入被用户中断")
        raise
    except Exception as e:
        logger.error(f"参数输入过程出错: {str(e)}")
        raise


class MicroneedleQualityModel(nn.Module):
    """微针质量预测模型"""

    def __init__(self, num_inputs, num_outputs, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()

        # 输入层
        prev_size = num_inputs
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(size))  # 添加批归一化
            self.layers.append(nn.Dropout(0.2))  # 添加dropout
            prev_size = size

        # 输出层
        self.layers.append(nn.Linear(prev_size, num_outputs))

        # 参数初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def generate_synthetic_data(num_samples: int = 1000, param_ranges: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成的训练数据

    Args:
        num_samples: 要生成的样本数量
        param_ranges: 可选的参数范围字典，如果为None则使用默认范围

    Returns:
        Tuple[np.ndarray, np.ndarray]: 训练数据和验证数据
    """
    try:
        # 使用提供的参数范围或默认范围
        ranges = param_ranges if param_ranges is not None else PARAM_RANGES

        # 生成随机参数
        print("开始生成随机参数...")
        X = []
        for i in range(num_samples):
            if i % 100 == 0:  # 每生成100个样本显示一次进度
                print(f"进度: {i}/{num_samples}")

            params = []
            for feature in input_features:
                min_val, max_val, step = ranges[feature]
                if feature in ['bottom_layers', 'print_mode']:
                    # 对于整数参数，使用整数步长
                    value = float(np.random.randint(min_val, max_val + 1))
                else:
                    # 对于连续参数，在范围内随机采样
                    value = np.random.uniform(min_val, max_val)
                params.append(value)
            X.append(params)
        X = np.array(X)
        print(f"参数生成完成，共 {len(X)} 组")

        # 生成对应的质量评分
        print("开始生成质量评分...")
        y = []
        for i, params in enumerate(X):
            if i % 100 == 0:  # 显示进度
                print(f"质量评分进度: {i}/{len(X)}")
            y.append(generate_quality_scores(params))
        y = np.array(y)
        print("质量评分生成完成")

        # 将数据集分为训练集和验证集
        print("划分训练集和验证集...")
        indices = np.random.permutation(num_samples)
        split_point = int(num_samples * 0.8)  # 80%用于训练

        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        train_data = np.column_stack((X[train_indices], y[train_indices]))
        val_data = np.column_stack((X[val_indices], y[val_indices]))

        # 创建数据目录（如果不存在）
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 保存数据集
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        header = ','.join(input_features + quality_score_components)

        train_file = os.path.join(data_dir, f'train_data_{timestamp}.csv')
        val_file = os.path.join(data_dir, f'val_data_{timestamp}.csv')

        print("保存数据到文件...")
        np.savetxt(train_file, train_data, delimiter=',', header=header, comments='')
        np.savetxt(val_file, val_data, delimiter=',', header=header, comments='')

        print(f"\n数据集生成完成！")
        print(f"训练集样本数: {len(train_data)}")
        print(f"验证集样本数: {len(val_data)}")
        print(f"\n文件已保存:")
        print(f"- 训练集: {train_file}")
        print(f"- 验证集: {val_file}")

        return train_data, val_data

    except Exception as e:
        logger.error(f"生成数据时出错: {str(e)}")
        print(f"生成数据时出错: {str(e)}")
        # 返回空数组，而不是None
        empty = np.empty((0, len(input_features) + len(quality_score_components)))
        return empty, empty


def generate_quality_scores(params):
    """基于输入参数生成质量评分"""
    try:
        # 获取打印方式并定义其影响因子
        print_mode = int(params[input_features.index('print_mode')])
        mode_factors = {
            0: {'definition': 1.0, 'adhesion': 0.9, 'height': 1.0},  # 连续打印
            1: {'definition': 0.95, 'adhesion': 1.0, 'height': 0.95},  # 间歇打印
            2: {'definition': 0.9, 'adhesion': 0.95, 'height': 0.9}  # 往复打印
        }
        mode_factor = mode_factors[print_mode]

        # 基础评分生成
        needle_def = (
                             0.4 * (params[input_features.index('exposure_time')] / PARAM_RANGES['exposure_time'][1]) +
                             0.3 * (params[input_features.index('exposure_intensity')] /
                                    PARAM_RANGES['exposure_intensity'][1]) +
                             0.2 * (1 - params[input_features.index('layer_height')] / PARAM_RANGES['layer_height'][
                         1]) +
                             0.1 * np.random.normal(0.8, 0.1)
                     ) * 100 * mode_factor['definition']

        layer_adh = (
                            0.3 * (1 - params[input_features.index('layer_height')] / PARAM_RANGES['layer_height'][1]) +
                            0.3 * (1 - params[input_features.index('lifting_speed')] / PARAM_RANGES['lifting_speed'][
                        1]) +
                            0.3 * (1 - params[input_features.index('liquid_speed')] / PARAM_RANGES['liquid_speed'][1]) +
                            0.1 * np.random.normal(0.8, 0.1)
                    ) * 100 * mode_factor['adhesion']

        needle_height = (
                                0.4 * (1 - params[input_features.index('layer_height')] / PARAM_RANGES['layer_height'][
                            1]) +
                                0.3 * (params[input_features.index('bottom_layers')] / PARAM_RANGES['bottom_layers'][
                            1]) +
                                0.3 * np.random.normal(0.8, 0.1)
                        ) * 100 * mode_factor['height']

        base_thick = (
                             0.4 * (params[input_features.index('bottom_exposure_time')] /
                                    PARAM_RANGES['bottom_exposure_time'][1]) +
                             0.4 * (params[input_features.index('bottom_exposure_intensity')] /
                                    PARAM_RANGES['bottom_exposure_intensity'][1]) +
                             0.2 * np.random.normal(0.8, 0.1)
                     ) * 100

        material_cure = (
                                0.3 * (
                                    params[input_features.index('exposure_time')] / PARAM_RANGES['exposure_time'][1]) +
                                0.3 * (params[input_features.index('exposure_intensity')] /
                                       PARAM_RANGES['exposure_intensity'][1]) +
                                0.2 * (params[input_features.index('exposure_wait')] / PARAM_RANGES['exposure_wait'][
                            1]) +
                                0.2 * np.random.normal(0.8, 0.1)
                        ) * 100

        # 限制所有分数在0-100范围内
        scores = [
            np.clip(needle_def, 0, 100),
            np.clip(layer_adh, 0, 100),
            np.clip(needle_height, 0, 100),
            np.clip(base_thick, 0, 100),
            np.clip(material_cure, 0, 100)
        ]

        return np.array(scores)

    except Exception as e:
        logger.error(f"生成质量评分时出错: {str(e)}")
        print(f"生成质量评分时出错: {str(e)}")
        # 返回零分评分，而不是None
        return np.zeros(len(quality_score_components))


def get_hyperparameters():
    """获取模型超参数"""
    logger.info("开始设置模型超参数")
    print("\n请输入模型超参数：")
    print("（直接按回车使用默认值）")

    try:
        # 获取层大小
        default_layers = [128, 128, 64]
        layer_input = safe_input(f"隐藏层大小（用逗号分隔）[{','.join(map(str, default_layers))}]: ")
        layer_sizes = [int(x.strip()) for x in layer_input.split(',')] if layer_input.strip() else default_layers

        # 获取训练参数
        epochs = safe_input("训练轮数 [500]: ")
        epochs = int(epochs) if epochs.strip() else 500

        batch_size = safe_input("批次大小 [32]: ")
        batch_size = int(batch_size) if batch_size.strip() else 32

        learning_rate = safe_input("学习率 [0.001]: ")
        learning_rate = float(learning_rate) if learning_rate.strip() else 0.001

        patience = safe_input("早停耐心值（轮数）[20]: ")
        patience = int(patience) if patience.strip() else 20

        hyperparams = {
            'layer_sizes': layer_sizes,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience
        }

        print("\n已选择的超参数：")
        for key, value in hyperparams.items():
            print(f"{key}: {value}")

        confirm = safe_input("\n使用这些参数继续？(y/n) [y]: ")
        if confirm.lower() == 'n':
            return get_hyperparameters()

        logger.info(f"超参数设置完成: {hyperparams}")
        return hyperparams

    except ValueError as e:
        logger.error(f"参数输入错误: {str(e)}")
        print(f"\n参数输入错误: {str(e)}")
        return get_hyperparameters()


def train_model(X_train, y_train, X_val, y_val, hyperparams):
    """训练模型"""
    logger.info("开始训练模型")
    try:
        # 准备数据加载器
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])

        # 创建模型
        model = MicroneedleQualityModel(len(input_features), len(quality_score_components),
                                        hyperparams['layer_sizes'])

        # 检查CUDA可用性并移动模型到适当的设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"模型已加载到设备: {device}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss = np.inf
        best_epoch = 0
        best_model_weights = None
        no_improve_count = 0

        print("\n训练进度：")
        for epoch in range(hyperparams['epochs']):
            try:
                # 训练阶段
                model.train()
                train_loss = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = inputs.float()
                    targets = targets.float()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # 评估阶段
                if (epoch + 1) % 10 == 0:  # 每10轮评估一次
                    metrics = evaluate_model(model, val_loader, device)
                    current_loss = metrics['total_mse']

                    # 学习率调整
                    scheduler.step(current_loss)
                    current_lr = optimizer.param_groups[0]['lr']

                    print(f"\n============ 第 {epoch + 1} 轮评估结果 ============")
                    print(f"训练损失: {train_loss:.4f}")
                    print(f"验证损失: {current_loss:.4f}")
                    print(f"当前学习率: {current_lr:.6f}")
                    print(f"总体评估等级: {metrics['quality_level']}")
                    print("各组件MSE损失:")
                    for comp, mse in metrics['component_mse'].items():
                        print(f"  {comp}: {mse:.4f}")
                    print(f"最大预测误差: {metrics['max_error']:.4f}")
                    print("=" * 45)

                    # 保存最佳模型
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_epoch = epoch
                        best_model_weights = model.state_dict()
                        no_improve_count = 0
                        logger.info(f"第 {epoch + 1} 轮更新最佳模型，验证损失: {best_loss:.4f}")
                    else:
                        no_improve_count += 1

                    # 早停检查
                    if no_improve_count >= hyperparams['patience']:
                        logger.info(f"触发早停机制，在第 {epoch + 1} 轮停止训练")
                        print(f"\n在轮次 {epoch + 1} 处提前停止")
                        print(f"最佳验证损失: {best_loss:.4f} (轮次 {best_epoch + 1})")
                        model.load_state_dict(best_model_weights)
                        break

            except KeyboardInterrupt:
                logger.info("训练被用户中断")
                print("\n训练被中断。保存当前最佳模型...")
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break
            except Exception as e:
                logger.error(f"训练过程中出错: {str(e)}")
                raise

        # 确保模型使用最佳权重
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        return model, best_loss

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise


def evaluate_model(model, val_loader, device):
    """详细评估模型性能"""
    model.eval()
    component_mse = {comp: [] for comp in quality_score_components}
    overall_predictions = []
    overall_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())

            # 计算每个组件的MSE
            for i, comp in enumerate(quality_score_components):
                mse = nn.MSELoss()(outputs[:, i], targets[:, i])
                component_mse[comp].append(mse.item())

            # 收集预测结果
            overall_predictions.extend(outputs.cpu().numpy())
            overall_targets.extend(targets.cpu().numpy())

    # 计算评估指标
    metrics = {
        'component_mse': {comp: np.mean(losses) for comp, losses in component_mse.items()},
        'total_mse': np.mean([np.mean(losses) for losses in component_mse.values()]),
        'max_error': np.max(np.abs(np.array(overall_predictions) - np.array(overall_targets)))
    }

    # 评估质量等级
    if metrics['total_mse'] < 25:
        quality_level = '优秀'
    elif metrics['total_mse'] < 100:
        quality_level = '良好'
    elif metrics['total_mse'] < 225:
        quality_level = '一般'
    else:
        quality_level = '需改进'

    metrics['quality_level'] = quality_level

    return metrics


def save_model(model: nn.Module, hyperparams: dict, loss: float, filename_prefix: str = 'model') -> None:
    """保存模型权重和配置信息"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保models目录存在
        if not os.path.exists('models'):
            os.makedirs('models')

        # 保存模型权重
        model_filename = os.path.join('models', f'{filename_prefix}_{timestamp}.pth')
        torch.save(model.state_dict(), model_filename)

        # 保存超参数和训练信息
        info = {
            'hyperparameters': hyperparams,
            'best_validation_loss': float(loss),
            'input_features': input_features,
            'quality_components': quality_score_components,
            'quality_weights': quality_score_weights,
            'param_ranges': PARAM_RANGES,
            'timestamp': timestamp,
            'model_file': os.path.basename(model_filename),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

        info_filename = os.path.join('models', f'{filename_prefix}_{timestamp}_info.json')
        with open(info_filename, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        logger.info(f"模型已保存至 {model_filename}")
        logger.info(f"模型信息已保存至 {info_filename}")

        print(f"\n模型已保存至 {model_filename}")
        print(f"模型信息已保存至 {info_filename}")

        return model_filename, info_filename

    except Exception as e:
        logger.error(f"保存模型时出错: {str(e)}")
        print(f"\n错误: 保存模型时出现问题: {str(e)}")
        return None, None

def predict_quality(model, params, device=None):
    """
    预测质量并提供可信度评估

    Args:
        model: 训练好的模型
        params: 输入参数字典
        device: 计算设备（如果为None则自动检测）

    Returns:
        dict: 包含预测结果和可信度的字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    try:
        # 准备输入数据
        input_vec = np.array([params[f] for f in input_features])
        input_tensor = torch.from_numpy(input_vec.astype(np.float32)).to(device)

        # 添加batch维度
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_vec = output_tensor.cpu().numpy()[0]  # 取出第一个样本的结果

        # 获取预测结果
        output_dict = {comp: output_vec[i] for i, comp in enumerate(quality_score_components)}

        # 计算综合质量分数
        overall_score = sum(output_dict[comp] * quality_score_weights[comp]
                            for comp in quality_score_components)

        # 评估预测可信度
        prediction_confidence = "高" if all(0 <= score <= 100 for score in output_vec) else "低"

        # 评估各组件预测可信度
        component_confidence = {}
        for comp, score in output_dict.items():
            if 0 <= score <= 100:
                if 20 <= score <= 90:  # 在一个更合理的范围内
                    component_confidence[comp] = "高"
                else:
                    component_confidence[comp] = "中"
            else:
                component_confidence[comp] = "低"

        # 生成参数优化建议
        optimization_suggestions = []
        for comp, score in output_dict.items():
            if score < 70:
                suggestions = get_optimization_suggestions(comp, params)
                optimization_suggestions.extend(suggestions)

        # 基于打印方式添加特殊提示
        mode_value = int(params['print_mode'])
        mode_tips = {
            0: "连续打印模式下建议注意:\n- 保持稳定的打印速度\n- 确保材料供应充足\n- 监控温度变化",
            1: "间歇打印模式下建议注意:\n- 控制等待时间\n- 确保层间结合\n- 避免过度冷却",
            2: "往复打印模式下建议注意:\n- 优化往复速度\n- 监控材料流动性\n- 注意表面质量"
        }

        return {
            'component_scores': output_dict,
            'overall_score': overall_score,
            'prediction_confidence': prediction_confidence,
            'component_confidence': component_confidence,
            'optimization_suggestions': optimization_suggestions,
            'print_mode_tip': mode_tips.get(mode_value, "未知打印模式"),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        raise


def get_optimization_suggestions(component: str, current_params: dict) -> List[str]:
    """根据组件得分生成优化建议"""
    suggestions = []

    if component == 'needle_definition':
        if current_params['exposure_time'] < PARAM_RANGES['exposure_time'][1] * 0.7:
            suggestions.append("考虑适当增加曝光时间")
        if current_params['exposure_intensity'] < PARAM_RANGES['exposure_intensity'][1] * 0.7:
            suggestions.append("可以提高曝光强度")
        if current_params['layer_height'] > PARAM_RANGES['layer_height'][1] * 0.3:
            suggestions.append("建议减小层高以提高精度")

    elif component == 'layer_adhesion':
        if current_params['lifting_speed'] > PARAM_RANGES['lifting_speed'][1] * 0.7:
            suggestions.append("建议降低抬升速度")
        if current_params['liquid_speed'] > PARAM_RANGES['liquid_speed'][1] * 0.7:
            suggestions.append("可以适当降低入液速度")

    elif component == 'needle_height':
        if current_params['bottom_layers'] < PARAM_RANGES['bottom_layers'][1] * 0.5:
            suggestions.append("可以增加底层数量")
        if current_params['layer_height'] > PARAM_RANGES['layer_height'][1] * 0.3:
            suggestions.append("考虑减小层高以提高精确度")

    elif component == 'base_thickness':
        if current_params['bottom_exposure_time'] < PARAM_RANGES['bottom_exposure_time'][1] * 0.7:
            suggestions.append("建议增加底层曝光时间")
        if current_params['bottom_exposure_intensity'] < PARAM_RANGES['bottom_exposure_intensity'][1] * 0.7:
            suggestions.append("可以提高底层曝光强度")

    elif component == 'material_curing':
        if current_params['exposure_time'] < PARAM_RANGES['exposure_time'][1] * 0.7:
            suggestions.append("建议增加曝光时间")
        if current_params['exposure_wait'] < PARAM_RANGES['exposure_wait'][1] * 0.5:
            suggestions.append("可以适当增加曝光等待时间")

    return suggestions


def find_optimal_parameters(model: nn.Module, num_combinations: int = 1000,
                            constraints: dict = None) -> Tuple[List[Dict], Dict]:
    """
    搜索最佳打印参数组合

    Args:
        model: 训练好的模型
        num_combinations: 要测试的参数组合数量
        constraints: 参数约束条件

    Returns:
        Tuple[List[Dict], Dict]: 最佳参数组合列表和参数分析结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def generate_param_combination():
        """生成一组随机参数组合"""
        params = {}
        for param, (min_val, max_val, step) in PARAM_RANGES.items():
            if constraints and param in constraints:
                # 使用约束条件中的范围
                c_min, c_max = constraints[param]
                min_val = max(min_val, c_min)
                max_val = min(max_val, c_max)

            if param in ['bottom_layers', 'print_mode']:
                possible_values = np.arange(min_val, max_val + step, step)
                params[param] = float(np.random.choice(possible_values))
            else:
                params[param] = np.random.uniform(min_val, max_val)
        return params

    def evaluate_combination(params: Dict) -> Tuple[float, str, Dict, Dict]:
        """评估一组参数组合的质量分数"""
        try:
            prediction_results = predict_quality(model, params, device)
            overall_score = prediction_results['overall_score']
            confidence = prediction_results['prediction_confidence']
            return overall_score, confidence, params, prediction_results
        except Exception as e:
            logger.error(f"评估参数组合时出错: {str(e)}")
            return -1, "低", params, None

    # 生成并评估参数组合
    logger.info("开始搜索最佳打印参数...")
    print("\n开始搜索最佳打印参数...")

    results = []
    progress_interval = max(1, num_combinations // 20)  # 5%的进度显示间隔

    for i in range(num_combinations):
        if (i + 1) % progress_interval == 0:
            progress = (i + 1) / num_combinations * 100
            print(f"搜索进度: {progress:.1f}% ({i + 1}/{num_combinations})")

        params = generate_param_combination()
        score, confidence, params, prediction_results = evaluate_combination(params)

        if score >= 0 and prediction_results:  # 只保存有效的结果
            results.append({
                'parameters': params,
                'score': score,
                'confidence': confidence,
                'component_scores': prediction_results['component_scores'],
                'optimization_suggestions': prediction_results['optimization_suggestions']
            })

    # 按分数排序并返回最佳结果
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_results = sorted_results[:5]  # 返回前5个最佳组合

    # 分析最佳参数的共同特征
    param_analysis = analyze_best_parameters(top_results)

    return top_results, param_analysis


def analyze_best_parameters(top_results: List[Dict]) -> Dict:
    """分析最佳参数组合的共同特征"""
    param_ranges = {param: [] for param in input_features}

    # 收集所有参数值
    for result in top_results:
        for param, value in result['parameters'].items():
            param_ranges[param].append(value)

    # 计算每个参数的统计信息
    analysis = {}
    for param, values in param_ranges.items():
        analysis[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

        # 添加参数趋势分析
        mean_value = analysis[param]['mean']
        param_range = PARAM_RANGES[param]
        range_center = (param_range[0] + param_range[1]) / 2

        if mean_value > range_center + param_range[2]:
            analysis[param]['trend'] = "偏高"
        elif mean_value < range_center - param_range[2]:
            analysis[param]['trend'] = "偏低"
        else:
            analysis[param]['trend'] = "适中"

    return analysis


def print_optimal_parameters(optimal_results: List[Dict], param_analysis: Dict):
    """打印最佳参数组合和分析结果"""
    print("\n=== 推荐的最佳打印参数 ===")
    print("\n注意：以下参数仅供参考，请结合实际情况调整")

    # 打印每组最佳参数
    for i, result in enumerate(optimal_results, 1):
        print(f"\n第 {i} 组推荐参数 (预期质量分数: {result['score']:.2f}, 预测可信度: {result['confidence']})")
        print("参数设置:")
        params = result['parameters']
        for param in input_features:
            value = params[param]

            # 根据参数类型调整显示格式
            if param == 'print_mode':
                formatted_value = f"{get_print_mode_name(int(value))}"
            elif param == 'bottom_layers':
                formatted_value = f"{int(value):d}"
            elif param in ['layer_height', 'lifting_speed', 'liquid_speed', 'exposure_wait']:
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value:.2f}"

            unit = get_parameter_unit(param)
            print(f"  {param}: {formatted_value}{unit}")

        print("\n质量评分详情:")
        for comp, score in result['component_scores'].items():
            print(f"  {comp}: {score:.2f}")

        if result['optimization_suggestions']:
            print("\n优化建议:")
            for suggestion in result['optimization_suggestions']:
                print(f"  - {suggestion}")

    # 打印参数分析结果
    print("\n=== 最佳参数分析 ===")
    print("参数趋势分析:")
    for param, analysis in param_analysis.items():
        unit = get_parameter_unit(param)
        if param == 'print_mode':
            continue  # 跳过打印模式的统计分析
        print(f"\n{param}:")
        print(f"  均值: {analysis['mean']:.3f}{unit}")
        print(f"  标准差: {analysis['std']:.3f}{unit}")
        print(f"  范围: {analysis['min']:.3f} - {analysis['max']:.3f}{unit}")
        print(f"  趋势: {analysis['trend']}")


def run_prediction_loop(model, device):
    """运行预测循环"""
    while True:
        try:
            params = input_params()
            prediction_results = predict_quality(model, params, device)

            print(f"\n=== 预测结果 ===")
            print(f"时间: {prediction_results['timestamp']}")
            print(f"\n总体质量分数: {prediction_results['overall_score']:.2f}")
            print(f"预测可信度: {prediction_results['prediction_confidence']}")

            print("\n打印模式提示:")
            print(prediction_results['print_mode_tip'])

            print("\n各组件分数:")
            for component, score in prediction_results['component_scores'].items():
                confidence = prediction_results['component_confidence'][component]
                print(f"{component}: {score:.2f} (可信度: {confidence})")

            if prediction_results['optimization_suggestions']:
                print("\n优化建议:")
                for suggestion in prediction_results['optimization_suggestions']:
                    print(f"- {suggestion}")

            print("\n是否要:")
            print("1. 继续预测")
            print("2. 保存该组参数")
            print("3. 返回主菜单")

            choice = safe_input("\n请选择 [1/2/3]: ")
            if choice == "2":
                # 保存参数到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'quality_prediction_{timestamp}.json'
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'parameters': params,
                        'prediction_results': prediction_results
                    }, f, indent=4, ensure_ascii=False)
                print(f"\n结果已保存至: {filename}")
            elif choice == "3":
                break

        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            print(f"\n预测过程中出错: {str(e)}")
            retry = safe_input("是否重试？(y/n) [y]: ")
            if retry.lower() == 'n':
                break


def load_latest_model(device):
    """加载最新的模型文件"""
    try:
        # 获取最新的模型文件
        model_files = sorted(
            [f for f in os.listdir() if f.startswith('microneedle_quality_model_')],
            reverse=True
        )
        if not model_files:
            raise FileNotFoundError("未找到模型文件")

        model_file = model_files[0]
        info_file = model_file.replace('model_', 'info_').replace('.pth', '.json')

        logger.info(f"加载模型: {model_file}")
        logger.info(f"加载模型信息: {info_file}")

        with open(info_file, 'r', encoding='utf-8') as f:
            model_info = json.load(f)

        model = MicroneedleQualityModel(
            len(input_features),
            len(quality_score_components),
            model_info['hyperparameters']['layer_sizes']
        )

        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        model.eval()

        return model, model_info

    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


def main():
    """主程序入口"""
    while True:
        try:
            print("\n=== 微针质量预测系统 ===")
            print("1. 训练新模型")
            print("2. 使用现有模型进行预测")
            print("3. 搜索最佳打印参数")
            print("4. 生成合成数据集")
            print("5. 查看参数说明")
            print("6. 退出")

            choice = safe_input("\n请选择操作 [1-6]: ")

            if choice == "1":
                print("\n=== 训练新模型 ===")
                print("1. 使用现有数据文件")
                print("2. 生成新的合成数据")
                data_choice = safe_input("请选择 [1/2]: ")

                if data_choice == "2":
                    num_samples = safe_input("请输入要生成的样本数量 [1000]: ")
                    num_samples = int(num_samples) if num_samples.strip() else 1000
                    train_data, val_data = generate_synthetic_data(num_samples)
                else:
                    print("\n加载训练数据...")
                    try:
                        data_files = sorted(
                            [f for f in os.listdir() if f.startswith('train_data_')],
                            reverse=True
                        )
                        if not data_files:
                            raise FileNotFoundError("未找到训练数据文件")

                        train_file = data_files[0]
                        val_file = train_file.replace('train_data_', 'val_data_')

                        logger.info(f"加载训练数据: {train_file}")
                        logger.info(f"加载验证数据: {val_file}")

                        train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
                        val_data = np.loadtxt(val_file, delimiter=',', skiprows=1)
                    except FileNotFoundError as e:
                        print(f"\n错误：{str(e)}。请先生成数据。")
                        continue
                    except Exception as e:
                        logger.error(f"加载数据时出错: {str(e)}")
                        print(f"\n加载数据时出错: {str(e)}")
                        continue

                X_train = train_data[:, :len(input_features)]
                y_train = train_data[:, len(input_features):]
                X_val = val_data[:, :len(input_features)]
                y_val = val_data[:, len(input_features):]

                print("数据加载成功！")
                print(f"训练集样本数: {len(X_train)}")
                print(f"验证集样本数: {len(X_val)}")

                hyperparams = get_hyperparameters()

                print("\n开始训练模型...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, best_loss = train_model(X_train, y_train, X_val, y_val, hyperparams)

                save_model(model, hyperparams, best_loss)
                print("\n模型训练完成！")

            elif choice == "2":
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, model_info = load_latest_model(device)
                    print("模型加载成功！")
                    print(f"模型信息:")
                    print(f"- 训练时间: {model_info['timestamp']}")
                    print(f"- 最佳验证损失: {model_info['best_validation_loss']:.4f}")

                    print("\n=== 参数范围参考 ===")
                    for param in input_features:
                        min_val, max_val, step = PARAM_RANGES[param]
                        unit = get_parameter_unit(param)
                        if param == 'print_mode':
                            print(f"{param}: 0-连续打印, 1-间歇打印, 2-往复打印")
                        else:
                            print(f"{param}: {min_val}-{max_val}{unit}, 步长{step}{unit}")

                    run_prediction_loop(model, device)

                except Exception as e:
                    logger.error(f"预测模式出错: {str(e)}")
                    print(f"\n预测模式出错: {str(e)}")
                    continue

            elif choice == "3":
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, _ = load_latest_model(device)
                    print("模型加载成功！")

                    # 设置搜索参数
                    print("\n搜索设置:")
                    num_combinations = safe_input("请输入要测试的参数组合数量 [1000]: ")
                    num_combinations = int(num_combinations) if num_combinations.strip() else 1000

                    print("\n是否要设置参数约束？")
                    print("1. 不设置约束")
                    print("2. 设置参数范围约束")
                    constraint_choice = safe_input("请选择 [1/2]: ")

                    constraints = None
                    if constraint_choice == "2":
                        constraints = {}
                        print("\n请输入需要约束的参数范围（直接回车跳过该参数）:")
                        for param in input_features:
                            min_val, max_val, _ = PARAM_RANGES[param]
                            unit = get_parameter_unit(param)

                            if param == 'print_mode':
                                print(f"\n当前参数: {param}")
                                print("0 - 连续打印")
                                print("1 - 间歇打印")
                                print("2 - 往复打印")
                                mode_input = safe_input("请输入允许的打印模式（用逗号分隔）[0,1,2]: ")
                                if mode_input.strip():
                                    allowed_modes = [int(x.strip()) for x in mode_input.split(',')]
                                    if allowed_modes:
                                        constraints[param] = (min(allowed_modes), max(allowed_modes))
                            else:
                                print(f"\n当前参数: {param}")
                                print(f"默认范围: {min_val}-{max_val}{unit}")
                                min_input = safe_input(f"最小值 [{min_val}]: ")
                                max_input = safe_input(f"最大值 [{max_val}]: ")

                                if min_input.strip() or max_input.strip():
                                    min_value = float(min_input) if min_input.strip() else min_val
                                    max_value = float(max_input) if max_input.strip() else max_val
                                    constraints[param] = (min_value, max_value)

                    # 搜索最佳参数
                    optimal_results, param_analysis = find_optimal_parameters(
                        model,
                        num_combinations=num_combinations,
                        constraints=constraints
                    )

                    # 打印结果
                    print_optimal_parameters(optimal_results, param_analysis)

                    # 保存结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'optimal_parameters_{timestamp}.json'
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump({
                            'optimal_results': optimal_results,
                            'param_analysis': param_analysis,
                            'constraints': constraints,
                            'search_settings': {
                                'num_combinations': num_combinations
                            }
                        }, f, indent=4, ensure_ascii=False)
                    print(f"\n结果已保存至: {filename}")

                except Exception as e:
                    logger.error(f"参数优化过程出错: {str(e)}")
                    print(f"\n参数优化过程出错: {str(e)}")
                    continue

            elif choice == "4":
                try:
                    print("\n=== 生成合成数据 ===")
                    print("此功能将生成用于模型训练的模拟数据")
                    print("数据生成基于预设的参数范围和质量影响规则")

                    num_samples = safe_input("请输入要生成的样本数量 [1000]: ")
                    num_samples = int(num_samples) if num_samples.strip() else 1000

                    print("\n生成数据设置:")
                    print("1. 均匀分布采样")
                    print("2. 聚焦于最佳参数区域采样")
                    sampling_choice = safe_input("请选择采样方式 [1/2]: ")

                    if sampling_choice == "2":
                        try:
                            # 读取最近的优化结果
                            optimal_files = sorted(
                                [f for f in os.listdir() if f.startswith('optimal_parameters_')],
                                reverse=True
                            )
                            if optimal_files:
                                with open(optimal_files[0], 'r', encoding='utf-8') as f:
                                    optimal_data = json.load(f)
                                param_analysis = optimal_data['param_analysis']

                                # 调整参数范围以聚焦于最佳区域
                                temp_ranges = PARAM_RANGES.copy()
                                for param, analysis in param_analysis.items():
                                    if param != 'print_mode':
                                        mean = analysis['mean']
                                        std = analysis['std']
                                        temp_ranges[param] = (
                                            max(PARAM_RANGES[param][0], mean - 2 * std),
                                            min(PARAM_RANGES[param][1], mean + 2 * std),
                                            PARAM_RANGES[param][2]
                                        )

                                print("\n使用优化后的参数范围生成数据...")
                                generate_synthetic_data(num_samples, param_ranges=temp_ranges)
                            else:
                                print("\n未找到优化结果，使用默认参数范围...")
                                generate_synthetic_data(num_samples)
                        except Exception as e:
                            logger.error(f"读取优化结果失败: {str(e)}")
                            print("\n读取优化结果失败，使用默认参数范围...")
                            generate_synthetic_data(num_samples)
                    else:
                        generate_synthetic_data(num_samples)

                except ValueError as e:
                    print(f"\n生成数据时出错: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"生成数据时出现意外错误: {str(e)}")
                    print(f"\n生成数据时出现意外错误: {str(e)}")
                    continue

            elif choice == "5":
                print("\n=== 参数说明 ===")
                print("\n打印参数说明:")
                for param in input_features:
                    min_val, max_val, step = PARAM_RANGES[param]
                    unit = get_parameter_unit(param)
                    print(f"\n{param}:")

                    if param == 'layer_height':
                        print("- 描述: 每层打印的厚度")
                        print("- 影响: 直接影响打印精度和打印时间")
                        print("- 建议: 精细结构选择较小值，粗糙结构可选择较大值")

                    elif param == 'exposure_time':
                        print("- 描述: 每层光固化的时间")
                        print("- 影响: 影响材料固化程度和打印速度")
                        print("- 建议: 根据材料特性和层厚调整")

                    elif param == 'exposure_intensity':
                        print("- 描述: 光源的输出强度")
                        print("- 影响: 影响固化效果和打印质量")
                        print("- 建议: 需要与曝光时间配合调整")

                    elif param == 'bottom_layers':
                        print("- 描述: 底部基础层的数量")
                        print("- 影响: 影响模型与基板的粘附强度")
                        print("- 建议: 根据模型重量和尺寸选择")

                    elif param == 'bottom_exposure_intensity':
                        print("- 描述: 底层的曝光强度")
                        print("- 影响: 影响模型与基板的粘附强度")
                        print("- 建议: 通常高于普通层的曝光强度")

                    elif param == 'lifting_speed':
                        print("- 描述: 打印平台抬升的速度")
                        print("- 影响: 影响层间分离效果和打印时间")
                        print("- 建议: 根据材料粘度调整")

                    elif param == 'exposure_wait':
                        print("- 描述: 曝光前的等待时间")
                        print("- 影响: 影响材料稳定性和均匀性")
                        print("- 建议: 根据材料流动性调整")

                    elif param == 'liquid_speed':
                        print("- 描述: 树脂流动填充速度")
                        print("- 影响: 影响层间材料填充均匀性")
                        print("- 建议: 根据材料粘度和模型结构调整")

                    elif param == 'bottom_exposure_time':
                        print("- 描述: 底层的曝光时间")
                        print("- 影响: 影响底层固化程度和基板附着力")
                        print("- 建议: 通常是普通层曝光时间的2-3倍")

                    elif param == 'print_mode':
                        print("- 描述: 打印模式选择")
                        print("- 选项: 0-连续打印, 1-间歇打印, 2-往复打印")
                        print("- 建议: ")
                        print("  * 连续打印: 适合简单结构，打印速度快")
                        print("  * 间歇打印: 适合精细结构，层间结合好")
                        print("  * 往复打印: 适合特殊结构，需要特别调整")

                    if param != 'print_mode':
                        print(f"- 取值范围: {min_val}-{max_val}{unit}")
                        print(f"- 调整步长: {step}{unit}")

                input("\n按回车键继续...")

            elif choice == "6":
                print("\n感谢使用！再见！")
                break

            else:
                print("\n无效的选择，请重试。")

        except ValueError as e:
            logger.error(f"输入错误: {str(e)}")
            print(f"\n输入错误: {str(e)}")
            retry = safe_input("是否继续？(y/n) [y]: ")
            if retry.lower() == 'n':
                break
        except Exception as e:
            logger.error(f"发生意外错误: {str(e)}")
            print(f"\n发生意外错误: {str(e)}")
            retry = safe_input("是否继续？(y/n) [y]: ")
            if retry.lower() == 'n':
                break

def ensure_data_structure():
    """确保数据目录结构正确"""
    directories = ['data', 'models', 'logs']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"创建目录: {dir_name}")

if __name__ == '__main__':
    try:
        print("\n=== 微针质量预测系统 ===")
        print("版本: 4.1.0")
        print("作者: CombjellyShen")
        print("日期: 2024-11-26")
        print("=" * 30)

        # 1. 检查依赖
        check_dependencies()
        ensure_data_structure()

        # 2. 打印系统信息
        try:
            print("\n=== 系统信息 ===")
            print(f"操作系统: {platform.system()} {platform.version()}")
            print(f"Python版本: {platform.python_version()}")
            print(f"CPU型号: {platform.processor()}")
            print(f"CPU核心数: {multiprocessing.cpu_count()}")

            if torch.cuda.is_available():
                print("\nGPU信息:")
                print(f"设备名称: {torch.cuda.get_device_name(0)}")
                print(f"CUDA版本: {torch.version.cuda}")
            else:
                print("\nCPU模式运行")
        except Exception as e:
            logger.warning(f"获取系统信息时出现警告: {str(e)}")

        # 3. 检查数据目录
        if not os.path.exists('data'):
            os.makedirs('data')
            logger.info("创建数据目录: data")

        # 4. 开始主程序循环
        while True:
            try:
                print("\n=== 主菜单 ===")
                print("1. 训练新模型")
                print("2. 使用现有模型进行预测")
                print("3. 搜索最佳打印参数")
                print("4. 生成合成数据集")
                print("5. 查看参数说明")
                print("6. 退出")

                choice = safe_input("\n请选择操作 [1-6]: ")

                if choice == "1":
                    print("\n=== 训练新模型 ===")
                    print("1. 使用现有数据文件")
                    print("2. 生成新的合成数据")
                    data_choice = safe_input("请选择 [1/2]: ")

                    if data_choice == "2":
                        num_samples = safe_input("请输入要生成的样本数量 [1000]: ")
                        num_samples = int(num_samples) if num_samples.strip() else 1000
                        generate_synthetic_data(num_samples)
                    elif data_choice == "1":
                        try:
                        # 检查现有数据文件
                            data_files = sorted([f for f in os.listdir('data') if f.startswith('train_data_')])
                            if not data_files:
                                print("\n错误: 未找到训练数据文件，请先生成数据。")
                                continue

                            print("\n找到以下数据文件:")
                            for i, f in enumerate(data_files, 1):
                                file_path = os.path.join('data', f)
                                file_size = os.path.getsize(file_path) / 1024  # 转换为KB
                                timestamp = f.split('_')[-1].replace('.csv', '')
                                # 尝试读取文件的样本数量
                                try:
                                    with open(file_path, 'r') as file:
                                        num_lines = sum(1 for line in file) - 1  # 减去头行
                                except:
                                    num_lines = "未知"
                                print(f"{i}. {f} (大小: {file_size:.1f}KB, 样本数: {num_lines})")

                            file_choice = safe_input("\n请选择要使用的数据文件 [1-{}]: ".format(len(data_files)))
                            try:
                                file_index = int(file_choice) - 1
                                if 0 <= file_index < len(data_files):
                                    selected_train_file = os.path.join('data', data_files[file_index])
                                    selected_val_file = selected_train_file.replace('train_data_', 'val_data_')

                                    if not os.path.exists(selected_val_file):
                                        print(f"\n错误: 未找到对应的验证数据文件: {os.path.basename(selected_val_file)}")
                                        continue

                                    print("\n加载数据文件...")
                                    train_data = np.loadtxt(selected_train_file, delimiter=',', skiprows=1)
                                    val_data = np.loadtxt(selected_val_file, delimiter=',', skiprows=1)

                                    print(f"数据加载成功!")
                                    print(f"训练集样本数: {len(train_data)}")
                                    print(f"验证集样本数: {len(val_data)}")

                                    # 分离特征和标签
                                    X_train = train_data[:, :len(input_features)]
                                    y_train = train_data[:, len(input_features):]
                                    X_val = val_data[:, :len(input_features)]
                                    y_val = val_data[:, len(input_features):]

                                    # 获取超参数并训练模型
                                    print("\n准备开始模型训练...")
                                    hyperparams = get_hyperparameters()

                                    print("\n开始训练模型...")
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                    model, best_loss = train_model(X_train, y_train, X_val, y_val, hyperparams)

                                    # 保存模型
                                    save_model(model, hyperparams, best_loss)
                                    print("\n模型训练完成！")

                                else:
                                    print("\n无效的选择！")
                                    continue
                            except ValueError:
                                print("\n请输入有效的数字！")
                                continue

                        except Exception as e:
                            logger.error(f"处理数据文件时出错: {str(e)}")
                            print(f"\n处理数据文件时出错: {str(e)}")
                            continue



                elif choice == "2":

                    try:

                        print("\n=== 使用模型预测 ===")

                        # 检查models目录

                        if not os.path.exists('models'):
                            print("\n错误: 未找到模型目录，请先训练模型（选项1）。")

                            continue

                        # 检查模型文件

                        model_files = []

                        for f in os.listdir('models'):

                            # 只查找pth文件

                            if f.endswith('.pth'):

                                info_file = f[:-4] + '_info.json'  # 移除.pth后缀并添加_info.json

                                info_path = os.path.join('models', info_file)

                                if os.path.exists(info_path):
                                    model_files.append((f, info_file))

                        if not model_files:
                            print("\n错误: 未找到已训练的模型文件，请先训练模型（选项1）。")

                            continue

                        print("\n找到以下模型文件:")

                        for i, (model_f, info_f) in enumerate(model_files, 1):

                            try:

                                with open(os.path.join('models', info_f), 'r', encoding='utf-8') as cf:

                                    config = json.load(cf)

                                    timestamp = config.get('timestamp', '未知')

                                    val_loss = config.get('best_validation_loss', '未知')

                                    print(f"{i}. {model_f}")

                                    print(f"   训练时间: {timestamp}")

                                    print(f"   验证损失: {val_loss:.4f}" if isinstance(val_loss, (
                                    int, float)) else f"   验证损失: {val_loss}")

                            except:

                                print(f"{i}. {model_f} (无法读取配置信息)")

                        model_choice = safe_input("\n请选择要使用的模型 [1-{}]: ".format(len(model_files)))

                        try:

                            model_index = int(model_choice) - 1

                            if 0 <= model_index < len(model_files):

                                model_f, info_f = model_files[model_index]

                                model_path = os.path.join('models', model_f)

                                info_path = os.path.join('models', info_f)

                                print("\n加载模型中...")

                                # 加载模型配置

                                with open(info_path, 'r', encoding='utf-8') as f:

                                    model_info = json.load(f)

                                # 创建模型实例

                                model = MicroneedleQualityModel(

                                    len(input_features),

                                    len(quality_score_components),

                                    model_info['hyperparameters']['layer_sizes']

                                )

                                # 设置设备

                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                # 加载模型权重

                                model.load_state_dict(torch.load(model_path, map_location=device))

                                model = model.to(device)

                                model.eval()

                                print("模型加载成功！")

                                print(f"\n模型信息:")

                                print(f"- 训练时间: {model_info['timestamp']}")

                                print(f"- 最佳验证损失: {model_info['best_validation_loss']:.4f}")

                                # 显示参数范围参考

                                print("\n=== 参数范围参考 ===")

                                for param in input_features:

                                    min_val, max_val, step = PARAM_RANGES[param]

                                    unit = get_parameter_unit(param)

                                    if param == 'print_mode':

                                        print(f"{param}: 0-连续打印, 1-间歇打印, 2-往复打印")

                                    else:

                                        print(f"{param}: {min_val}-{max_val}{unit}, 步长{step}{unit}")

                                # 运行预测循环

                                run_prediction_loop(model, device)


                            else:

                                print("\n无效的选择！")

                                continue


                        except ValueError:

                            print("\n请输入有效的数字！")

                            continue


                    except Exception as e:

                        logger.error(f"加载模型时出错: {str(e)}")

                        print(f"\n加载模型时出错: {str(e)}")

                        continue


                elif choice == "3":

                    try:

                        print("\n=== 搜索最佳参数 ===")

                        # 检查models目录

                        if not os.path.exists('models'):
                            print("\n错误: 未找到模型目录，请先训练模型（选项1）。")

                            continue

                        # 检查模型文件

                        model_files = []

                        for f in os.listdir('models'):

                            if f.endswith('.pth'):

                                info_file = f[:-4] + '_info.json'

                                info_path = os.path.join('models', info_file)

                                if os.path.exists(info_path):
                                    model_files.append((f, info_file))

                        if not model_files:
                            print("\n错误: 未找到已训练的模型文件，请先训练模型（选项1）。")

                            continue

                        print("\n找到以下模型文件:")

                        for i, (model_f, info_f) in enumerate(model_files, 1):

                            try:

                                with open(os.path.join('models', info_f), 'r', encoding='utf-8') as cf:

                                    config = json.load(cf)

                                    timestamp = config.get('timestamp', '未知')

                                    val_loss = config.get('best_validation_loss', '未知')

                                    print(f"{i}. {model_f}")

                                    print(f"   训练时间: {timestamp}")

                                    print(f"   验证损失: {val_loss:.4f}" if isinstance(val_loss, (
                                    int, float)) else f"   验证损失: {val_loss}")

                            except:

                                print(f"{i}. {model_f} (无法读取配置信息)")

                        model_choice = safe_input("\n请选择要使用的模型 [1-{}]: ".format(len(model_files)))

                        try:

                            model_index = int(model_choice) - 1

                            if 0 <= model_index < len(model_files):

                                model_f, info_f = model_files[model_index]

                                model_path = os.path.join('models', model_f)

                                info_path = os.path.join('models', info_f)

                                print("\n加载模型中...")

                                # 加载模型配置

                                with open(info_path, 'r', encoding='utf-8') as f:

                                    model_info = json.load(f)

                                # 创建模型实例

                                model = MicroneedleQualityModel(

                                    len(input_features),

                                    len(quality_score_components),

                                    model_info['hyperparameters']['layer_sizes']

                                )

                                # 设置设备

                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                                # 加载模型权重

                                model.load_state_dict(torch.load(model_path, map_location=device))

                                model = model.to(device)

                                model.eval()

                                print("模型加载成功！")

                                # 设置搜索参数

                                print("\n搜索设置:")

                                num_combinations = safe_input("请输入要测试的参数组合数量 [1000]: ")

                                num_combinations = int(num_combinations) if num_combinations.strip() else 1000

                                print("\n是否要设置参数约束？")

                                print("1. 不设置约束")

                                print("2. 设置参数范围约束")

                                constraint_choice = safe_input("请选择 [1/2]: ")

                                constraints = None

                                if constraint_choice == "2":

                                    constraints = {}

                                    print("\n请输入需要约束的参数范围（直接回车跳过该参数）:")

                                    for param in input_features:

                                        min_val, max_val, _ = PARAM_RANGES[param]

                                        unit = get_parameter_unit(param)

                                        if param == 'print_mode':

                                            print(f"\n当前参数: {param}")

                                            print("0 - 连续打印")

                                            print("1 - 间歇打印")

                                            print("2 - 往复打印")

                                            mode_input = safe_input("请输入允许的打印模式（用逗号分隔）[0,1,2]: ")

                                            if mode_input.strip():

                                                allowed_modes = [int(x.strip()) for x in mode_input.split(',')]

                                                if allowed_modes:
                                                    constraints[param] = (min(allowed_modes), max(allowed_modes))

                                        else:

                                            print(f"\n当前参数: {param}")

                                            print(f"默认范围: {min_val}-{max_val}{unit}")

                                            min_input = safe_input(f"最小值 [{min_val}]: ")

                                            max_input = safe_input(f"最大值 [{max_val}]: ")

                                            if min_input.strip() or max_input.strip():
                                                min_value = float(min_input) if min_input.strip() else min_val

                                                max_value = float(max_input) if max_input.strip() else max_val

                                                constraints[param] = (min_value, max_value)

                                # 搜索最佳参数

                                print("\n开始搜索最佳参数组合...")

                                optimal_results, param_analysis = find_optimal_parameters(

                                    model,

                                    num_combinations=num_combinations,

                                    constraints=constraints

                                )

                                # 打印结果

                                print_optimal_parameters(optimal_results, param_analysis)

                                # 保存结果

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                                results_dir = 'results'

                                if not os.path.exists(results_dir):
                                    os.makedirs(results_dir)

                                filename = os.path.join(results_dir, f'optimal_parameters_{timestamp}.json')

                                with open(filename, 'w', encoding='utf-8') as f:

                                    json.dump({

                                        'optimal_results': optimal_results,

                                        'param_analysis': param_analysis,

                                        'constraints': constraints,

                                        'search_settings': {

                                            'num_combinations': num_combinations,

                                            'model_used': model_f,

                                            'search_time': timestamp

                                        }

                                    }, f, indent=4, ensure_ascii=False)

                                print(f"\n结果已保存至: {filename}")


                            else:

                                print("\n无效的选择！")

                                continue


                        except ValueError:

                            print("\n请输入有效的数字！")

                            continue


                    except Exception as e:

                        logger.error(f"参数优化过程出错: {str(e)}")

                        print(f"\n参数优化过程出错: {str(e)}")

                        continue


                elif choice == "4":

                    print("\n=== 生成合成数据 ===")

                    try:

                        num_samples = safe_input("请输入要生成的样本数量 [1000]: ")

                        num_samples = int(num_samples) if num_samples.strip() else 1000

                        print("\n开始生成数据...")

                        train_data, val_data = generate_synthetic_data(num_samples)

                        if len(train_data) > 0:

                            print("\n数据生成成功！")

                        else:

                            print("\n数据生成失败，请检查错误信息。")


                    except ValueError as e:

                        print(f"\n输入错误: {str(e)}")

                    except Exception as e:

                        print(f"\n生成数据时出错: {str(e)}")

                elif choice == "5":
                    print("\n=== 参数说明 ===")
                    for param in input_features:
                        min_val, max_val, step = PARAM_RANGES[param]
                        unit = get_parameter_unit(param)
                        print(f"\n{param}:")
                        print(f"范围: {min_val}-{max_val}{unit}")
                        print(f"步长: {step}{unit}")
                    input("\n按回车键继续...")

                elif choice == "6":
                    print("\n感谢使用！再见！")
                    break

                else:
                    print("\n无效的选择，请重试。")

            except ValueError as e:
                print(f"\n输入错误: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"操作出错: {str(e)}")
                print(f"\n操作出错: {str(e)}")
                retry = safe_input("\n是否继续？(y/n) [y]: ")
                if retry.lower() == 'n':
                    break

    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        logger.error(f"程序出错: {str(e)}")
        print(f"\n程序出错: {str(e)}")
    finally:
        print("\n正在退出程序...")
        logging.shutdown()