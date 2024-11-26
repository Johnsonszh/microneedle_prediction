import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import json
import sys
from typing import Dict, List, Tuple

# 定义输入特征和质量评分组件
input_features = [
    'exposure_time', 'exposure_intensity', 'layer_height',
    'lifting_speed', 'exposure_wait', 'bottom_layers',
    'bottom_exposure_time', 'print_temperature'
]

quality_score_components = [
    'needle_definition', 'layer_adhesion', 'needle_height',
    'base_thickness', 'material_curing'
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
    'exposure_time': (1.0, 10.0, 0.5),  # 范围：1-10秒，步长0.5
    'exposure_intensity': (50, 100, 5),  # 范围：50-100%，步长5
    'layer_height': (0.02, 0.1, 0.01),  # 范围：0.02-0.1mm，步长0.01
    'lifting_speed': (0.5, 3.0, 0.5),  # 范围：0.5-3.0mm/s，步长0.5
    'exposure_wait': (0.1, 2.0, 0.1),  # 范围：0.1-2.0秒，步长0.1
    'bottom_layers': (3, 8, 1),  # 范围：3-8层，步长1
    'bottom_exposure_time': (10, 30, 2),  # 范围：10-30秒，步长2
    'print_temperature': (20, 30, 1)  # 范围：20-30℃，步长1
}


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
            prev_size = size

        # 输出层
        self.layers.append(nn.Linear(prev_size, num_outputs))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
        'exposure_time': '秒',
        'exposure_intensity': '%',
        'layer_height': 'mm',
        'lifting_speed': 'mm/s',
        'exposure_wait': '秒',
        'bottom_layers': '层',
        'bottom_exposure_time': '秒',
        'print_temperature': '℃'
    }
    return units.get(param, '')


def generate_synthetic_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成的训练数据和验证数据

    Args:
        num_samples: 要生成的样本数量

    Returns:
        Tuple[np.ndarray, np.ndarray]: 训练数据和验证数据
    """

    def generate_quality_scores(params):
        """基于输入参数生成质量评分"""
        scores = []

        # 针尖清晰度 (exposure_time 和 exposure_intensity 的影响较大)
        needle_def = (
                             0.4 * (params[0] / PARAM_RANGES['exposure_time'][1]) +  # exposure_time
                             0.4 * (params[1] / PARAM_RANGES['exposure_intensity'][1]) +  # exposure_intensity
                             0.2 * np.random.normal(0.8, 0.1)  # 随机因素
                     ) * 100

        # 层间粘结性 (layer_height 和 lifting_speed 的影响较大)
        layer_adh = (
                            0.4 * (1 - params[2] / PARAM_RANGES['layer_height'][1]) +  # layer_height
                            0.3 * (1 - params[3] / PARAM_RANGES['lifting_speed'][1]) +  # lifting_speed
                            0.3 * np.random.normal(0.8, 0.1)  # 随机因素
                    ) * 100

        # 针尖高度 (layer_height 和 bottom_layers 的影响较大)
        needle_height = (
                                0.35 * (1 - params[2] / PARAM_RANGES['layer_height'][1]) +  # layer_height
                                0.35 * (params[5] / PARAM_RANGES['bottom_layers'][1]) +  # bottom_layers
                                0.3 * np.random.normal(0.8, 0.1)  # 随机因素
                        ) * 100

        # 底座厚度 (bottom_exposure_time 的影响较大)
        base_thick = (
                             0.6 * (params[6] / PARAM_RANGES['bottom_exposure_time'][1]) +  # bottom_exposure_time
                             0.4 * np.random.normal(0.8, 0.1)  # 随机因素
                     ) * 100

        # 材料固化度 (exposure_time, temperature 和 exposure_wait 的影响较大)
        material_cure = (
                                0.3 * (params[0] / PARAM_RANGES['exposure_time'][1]) +  # exposure_time
                                0.3 * (1 - (params[7] - 20) / 10) +  # temperature (20-30℃)
                                0.2 * (params[4] / PARAM_RANGES['exposure_wait'][1]) +  # exposure_wait
                                0.2 * np.random.normal(0.8, 0.1)  # 随机因素
                        ) * 100

        # 将所有分数限制在0-100范围内
        scores = [
            np.clip(needle_def, 0, 100),
            np.clip(layer_adh, 0, 100),
            np.clip(needle_height, 0, 100),
            np.clip(base_thick, 0, 100),
            np.clip(material_cure, 0, 100)
        ]

        return np.array(scores)

    # 生成随机参数
    print("生成随机参数...")
    X = []
    for _ in range(num_samples):
        params = []
        for feature in input_features:
            min_val, max_val, step = PARAM_RANGES[feature]
            if feature == 'bottom_layers':
                # 对于整数参数，使用整数步长
                value = float(np.random.randint(min_val, max_val + 1))
            else:
                # 对于连续参数，在范围内随机采样
                value = np.random.uniform(min_val, max_val)
            params.append(value)
        X.append(params)
    X = np.array(X)

    # 生成对应的质量评分
    print("生成质量评分...")
    y = np.array([generate_quality_scores(params) for params in X])

    # 将数据集分为训练集和验证集
    print("划分训练集和验证集...")
    indices = np.random.permutation(num_samples)
    split_point = int(num_samples * 0.8)  # 80%用于训练

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_data = np.column_stack((X[train_indices], y[train_indices]))
    val_data = np.column_stack((X[val_indices], y[val_indices]))

    # 保存数据集
    print("保存数据集...")
    header = ','.join(input_features + quality_score_components)
    np.savetxt('train_data.csv', train_data, delimiter=',', header=header, comments='')
    np.savetxt('val_data.csv', val_data, delimiter=',', header=header, comments='')

    print(f"\n数据集生成完成！")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")

    return train_data, val_data


def get_hyperparameters():
    """获取模型超参数"""
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

        return hyperparams

    except ValueError as e:
        print(f"\n参数输入错误: {str(e)}")
        return get_hyperparameters()


def save_model(model: nn.Module, hyperparams: dict, loss: float, filename_prefix: str = 'microneedle_quality') -> None:
    """保存模型权重和配置信息"""
    try:
        # 保存模型权重
        model_filename = f'{filename_prefix}_model.pth'
        torch.save(model.state_dict(), model_filename)

        # 保存超参数和训练信息
        info = {
            'hyperparameters': hyperparams,
            'best_validation_loss': float(loss),
            'input_features': input_features,
            'quality_components': quality_score_components,
            'quality_weights': quality_score_weights
        }

        info_filename = f'{filename_prefix}_info.json'
        with open(info_filename, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        print(f"\n模型已保存至 {model_filename}")
        print(f"模型信息已保存至 {info_filename}")

    except Exception as e:
        print(f"\n错误: 保存模型时出现问题: {str(e)}")


def evaluate_model(model, val_loader):
    """详细评估模型性能"""
    model.eval()
    component_mse = {comp: [] for comp in quality_score_components}
    overall_predictions = []
    overall_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.float())

            # 计算每个组件的MSE
            for i, comp in enumerate(quality_score_components):
                mse = nn.MSELoss()(outputs[:, i], targets[:, i])
                component_mse[comp].append(mse.item())

            # 存储预测值和真实值用于计算整体指标
            overall_predictions.extend(outputs.numpy())
            overall_targets.extend(targets.numpy())

    # 计算平均指标
    metrics = {
        'component_mse': {comp: np.mean(losses) for comp, losses in component_mse.items()},
        'total_mse': np.mean([np.mean(losses) for losses in component_mse.values()]),
        'max_error': np.max(np.abs(np.array(overall_predictions) - np.array(overall_targets)))
    }

    # 添加评估结果解释
    quality_level = ''
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


def train_model(X_train, y_train, X_val, y_val, hyperparams):
    """训练模型"""
    try:
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])

        model = MicroneedleQualityModel(len(input_features), len(quality_score_components),
                                        hyperparams['layer_sizes'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

        best_loss = np.inf
        best_epoch = 0
        best_model_weights = None

        print("\n训练进度：")
        for epoch in range(hyperparams['epochs']):
            try:
                model.train()
                for inputs, targets in train_loader:
                    outputs = model(inputs.float())
                    loss = criterion(outputs, targets.float())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 每50轮进行详细评估
                if (epoch + 1) % 50 == 0:
                    metrics = evaluate_model(model, val_loader)
                    print(f"\n============ 第 {epoch + 1} 轮评估结果 ============")
                    print(f"总体评估等级: {metrics['quality_level']}")
                    print(f"总体MSE损失: {metrics['total_mse']:.4f}")
                    print("各组件MSE损失:")
                    for comp, mse in metrics['component_mse'].items():
                        print(f"  {comp}: {mse:.4f}")
                    print(f"最大预测误差: {metrics['max_error']:.4f}")
                    print("=" * 45)

                    current_loss = metrics['total_mse']
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_epoch = epoch
                        best_model_weights = model.state_dict()
                    elif epoch - best_epoch >= hyperparams['patience']:
                        print(f"\n在轮次 {epoch + 1} 处提前停止")
                        print(f"最佳验证损失: {best_loss:.4f} (轮次 {best_epoch + 1})")
                        model.load_state_dict(best_model_weights)
                        break

            except KeyboardInterrupt:
                print("\n训练被中断。保存当前最佳模型...")
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break

        return model, best_loss

    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        raise


def predict_quality(model, params):
    """
    预测质量并提供可信度评估

    Args:
        model: 训练好的模型
        params: 输入参数字典

    Returns:
        dict: 包含预测结果和可信度的字典
    """
    model.eval()
    try:
        input_vec = np.array([params[f] for f in input_features])

        with torch.no_grad():
            input_tensor = torch.from_numpy(input_vec.astype(np.float32))
            output_tensor = model(input_tensor)

        output_vec = output_tensor.numpy()
        output_dict = {comp: output_vec[i] for i, comp in enumerate(quality_score_components)}

        # 计算综合质量分数
        overall_score = sum(output_dict[comp] * quality_score_weights[comp]
                            for comp in quality_score_components)

        # 添加预测可信度评估
        prediction_confidence = "高" if 0 <= overall_score <= 100 else "低"

        component_confidence = {}
        for comp, score in output_dict.items():
            if 0 <= score <= 100:
                component_confidence[comp] = "高"
            else:
                component_confidence[comp] = "低"

        return {
            'component_scores': output_dict,
            'overall_score': overall_score,
            'prediction_confidence': prediction_confidence,
            'component_confidence': component_confidence
        }

    except Exception as e:
        print(f"\n预测过程中出错: {str(e)}")
        raise


def find_optimal_parameters(model: nn.Module, num_combinations: int = 1000) -> List[Dict]:
    """
    搜索最佳打印参数组合

    Args:
        model: 训练好的模型
        num_combinations: 要测试的参数组合数量

    Returns:
        最佳参数组合列表（按质量评分排序）
    """

    def generate_param_combination():
        """生成一组随机参数组合"""
        params = {}
        for param, (min_val, max_val, step) in PARAM_RANGES.items():
            if param == 'bottom_layers':
                # 对于整数参数，使用整数步长
                possible_values = np.arange(min_val, max_val + step, step)
                params[param] = float(np.random.choice(possible_values))
            else:
                # 对于连续参数，在范围内随机采样
                params[param] = np.random.uniform(min_val, max_val)
        return params

    def evaluate_combination(params: Dict) -> Tuple[float, Dict]:
        """评估一组参数组合的质量分数"""
        try:
            prediction_results = predict_quality(model, params)
            overall_score = prediction_results['overall_score']
            confidence = prediction_results['prediction_confidence']
            return overall_score, confidence, params
        except Exception as e:
            print(f"评估参数组合时出错: {str(e)}")
            return -1, "低", params

    # 生成并评估参数组合
    print("\n开始搜索最佳打印参数...")
    results = []
    for i in range(num_combinations):
        if (i + 1) % 100 == 0:
            print(f"已评估 {i + 1}/{num_combinations} 组参数组合")

        params = generate_param_combination()
        score, confidence, params = evaluate_combination(params)

        if score >= 0:  # 只保存有效的结果
            results.append({
                'parameters': params,
                'score': score,
                'confidence': confidence
            })

    # 按分数排序并返回最佳结果
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_results[:5]  # 返回前5个最佳组合


def print_optimal_parameters(optimal_results: List[Dict]):
    """
    打印最佳参数组合

    Args:
        optimal_results: 最佳参数组合列表
    """
    print("\n=== 推荐的最佳打印参数 ===")
    print("\n注意：以下参数仅供参考，请结合实际情况调整")

    for i, result in enumerate(optimal_results, 1):
        print(f"\n第 {i} 组推荐参数 (预期质量分数: {result['score']:.2f}, 预测可信度: {result['confidence']})")
        print("参数设置:")
        for param, value in result['parameters'].items():
            # 根据参数类型调整显示格式
            if param == 'bottom_layers':
                formatted_value = f"{int(value):d}"
            elif param in ['layer_height', 'lifting_speed', 'exposure_wait']:
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value:.2f}"

            # 添加单位
            unit = get_parameter_unit(param)
            print(f"  {param}: {formatted_value}{unit}")


def input_params():
    """获取用户输入的参数值"""
    params = {}
    print("\n请输入以下参数：")
    for feature in input_features:
        while True:
            try:
                value_str = safe_input(f"{feature} [{get_parameter_unit(feature)}]: ")
                value = float(value_str)
                params[feature] = value
                break
            except ValueError:
                print("请输入有效的数字")
    return params


def run_prediction_loop(model):
    """运行预测循环"""
    while True:
        try:
            params = input_params()
            prediction_results = predict_quality(model, params)

            print(f"\n预测的总体质量分数: {prediction_results['overall_score']:.2f}")
            print(f"预测可信度: {prediction_results['prediction_confidence']}")
            print("\n各组件分数:")
            for component, score in prediction_results['component_scores'].items():
                confidence = prediction_results['component_confidence'][component]
                print(f"{component}: {score:.2f} (可信度: {confidence})")

            repeat = safe_input("\n是否继续预测？(y/n) [y]: ")
            if repeat.lower() == 'n':
                break

        except Exception as e:
            print(f"\n预测过程中出错: {str(e)}")
            retry = safe_input("是否重试？(y/n) [y]: ")
            if retry.lower() == 'n':
                break


def main():
    """主程序入口"""
    while True:
        try:
            print("\n=== 微针质量预测系统 ===")
            print("1. 训练新模型")
            print("2. 使用现有模型进行预测")
            print("3. 搜索最佳打印参数")
            print("4. 生成合成数据集")
            print("5. 退出")

            choice = safe_input("\n请选择操作 [1/2/3/4/5]: ")

            if choice == "1":
                print("\n是否需要生成新的训练数据？")
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
                        train_data = np.loadtxt('train_data.csv', delimiter=',', skiprows=1)
                        val_data = np.loadtxt('val_data.csv', delimiter=',', skiprows=1)
                    except FileNotFoundError:
                        print("\n错误：未找到训练数据文件。请先生成数据或确保数据文件存在。")
                        continue

                X_train = train_data[:, :len(input_features)]
                y_train = train_data[:, len(input_features):]
                X_val = val_data[:, :len(input_features)]
                y_val = val_data[:, len(input_features):]

                print("数据加载成功！")

                hyperparams = get_hyperparameters()

                print("\n开始训练模型...")
                model, best_loss = train_model(X_train, y_train, X_val, y_val, hyperparams)

                save_model(model, hyperparams, best_loss)
                print("\n模型训练完成！")

            elif choice == "2":
                try:
                    with open('microneedle_quality_info.json', 'r', encoding='utf-8') as f:
                        model_info = json.load(f)

                    model = MicroneedleQualityModel(
                        len(input_features),
                        len(quality_score_components),
                        model_info['hyperparameters']['layer_sizes']
                    )

                    model.load_state_dict(torch.load('microneedle_quality_model.pth'))
                    model.eval()
                    print("\n模型加载成功！")

                    run_prediction_loop(model)

                except FileNotFoundError:
                    print("\n错误：未找到模型文件。请先训练模型。")
                    continue

            elif choice == "3":
                try:
                    # 加载模型
                    with open('microneedle_quality_info.json', 'r', encoding='utf-8') as f:
                        model_info = json.load(f)

                    model = MicroneedleQualityModel(
                        len(input_features),
                        len(quality_score_components),
                        model_info['hyperparameters']['layer_sizes']
                    )

                    model.load_state_dict(torch.load('microneedle_quality_model.pth'))
                    model.eval()
                    print("\n模型加载成功！")

                    # 获取用户输入的搜索参数
                    num_combinations = safe_input("请输入要测试的参数组合数量 [1000]: ")
                    num_combinations = int(num_combinations) if num_combinations.strip() else 1000

                    # 搜索最佳参数
                    optimal_results = find_optimal_parameters(model, num_combinations)
                    print_optimal_parameters(optimal_results)

                except FileNotFoundError:
                    print("\n错误：未找到模型文件。请先训练模型。")
                    continue
                except Exception as e:
                    print(f"\n搜索最佳参数时出错: {str(e)}")
                    continue

            elif choice == "4":
                try:
                    num_samples = safe_input("请输入要生成的样本数量 [1000]: ")
                    num_samples = int(num_samples) if num_samples.strip() else 1000
                    generate_synthetic_data(num_samples)
                except ValueError as e:
                    print(f"\n生成数据时出错: {str(e)}")
                    continue
                except Exception as e:
                    print(f"\n生成数据时出现意外错误: {str(e)}")
                    continue

            elif choice == "5":
                print("\n感谢使用！再见！")
                break

            else:
                print("\n无效的选择，请重试。")

        except ValueError as e:
            print(f"\n输入错误: {str(e)}")
            retry = safe_input("是否继续？(y/n) [y]: ")
            if retry.lower() == 'n':
                break
        except Exception as e:
            print(f"\n发生意外错误: {str(e)}")
            retry = safe_input("是否继续？(y/n) [y]: ")
            if retry.lower() == 'n':
                break


def check_cuda_availability():
    """检查CUDA是否可用并打印设备信息"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nCUDA可用！使用设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\nCUDA不可用，使用CPU进行计算")
    return device


if __name__ == '__main__':
    try:
        # 检查CUDA可用性
        device = check_cuda_availability()

        # 打印系统信息
        print("\n=== 微针质量预测系统 ===")
        print("版本: 3.0.0")
        print("作者: CombjellyShen")
        print("日期: 2024-11-18")
        print("=" * 30)

        # 运行主程序
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断。正在退出...")
    except Exception as e:
        print(f"\n严重错误: {str(e)}")
    finally:
        print("\n程序已终止。")