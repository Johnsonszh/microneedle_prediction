import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import json
import sys

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


class MicroneedleQualityModel(nn.Module):
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


def save_model(model: nn.Module, hyperparams: dict, loss: float, filename_prefix: str = 'microneedle_quality') -> None:
    """
    保存模型权重和配置信息。

    参数:
        model: 训练好的PyTorch模型
        hyperparams: 包含模型超参数的字典
        loss: 训练过程中达到的最佳验证损失
        filename_prefix: 保存文件的前缀
    """
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


def get_hyperparameters():
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


def train_model(X_train, y_train, X_val, y_val, hyperparams):
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

                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for val_inputs, val_targets in val_loader:
                        val_outputs = model(val_inputs.float())
                        val_loss = criterion(val_outputs, val_targets.float())
                        val_losses.append(val_loss.item())

                current_loss = np.mean(val_losses)
                if (epoch + 1) % 50 == 0:
                    print(f"轮次 {epoch + 1}/{hyperparams['epochs']}, 验证损失: {current_loss:.4f}")

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
    model.eval()
    try:
        input_vec = np.array([params[f] for f in input_features])

        with torch.no_grad():
            input_tensor = torch.from_numpy(input_vec.astype(np.float32))
            output_tensor = model(input_tensor)

        output_vec = output_tensor.numpy()
        output_dict = {comp: output_vec[i] for i, comp in enumerate(quality_score_components)}

        overall_score = sum(output_dict[comp] * quality_score_weights[comp]
                            for comp in quality_score_components)

        return output_dict, overall_score

    except Exception as e:
        print(f"\n预测过程中出错: {str(e)}")
        raise


def input_params():
    params = {}
    print("\n请输入以下参数：")
    for feature in input_features:
        while True:
            try:
                value_str = safe_input(f"{feature}: ")
                value = float(value_str)
                params[feature] = value
                break
            except ValueError:
                print("请输入有效的数字")
    return params


def run_prediction_loop(model):
    while True:
        try:
            params = input_params()
            component_scores, overall_score = predict_quality(model, params)

            print(f"\n预测的总体质量分数: {overall_score:.2f}")
            print("各组件分数：")
            for component, score in component_scores.items():
                print(f"{component}: {score:.2f}")

            repeat = safe_input("\n是否继续预测？(y/n) [y]: ")
            if repeat.lower() == 'n':
                break

        except Exception as e:
            print(f"\n预测过程中出错: {str(e)}")
            retry = safe_input("是否重试？(y/n) [y]: ")
            if retry.lower() == 'n':
                break


def main():
    while True:
        try:
            print("\n=== 微针质量预测系统 ===")
            print("1. 训练新模型")
            print("2. 使用现有模型进行预测")
            print("3. 退出")

            choice = safe_input("\n请选择操作 [1/2/3]: ")

            if choice == "1":
                print("\n加载训练数据...")
                try:
                    train_data = np.loadtxt('.venv/train_data.csv', delimiter=',', skiprows=1)
                    val_data = np.loadtxt('.venv/val_data.csv', delimiter=',', skiprows=1)

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

                except FileNotFoundError:
                    print("\n错误：未找到训练数据文件。请确保 train_data.csv 和 val_data.csv 文件存在。")
                    continue

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
                print("\n感谢使用！再见！")
                break

            else:
                print("\n无效的选择，请重试。")

        except Exception as e:
            print(f"\n发生意外错误: {str(e)}")
            retry = safe_input("是否继续？(y/n) [y]: ")
            if retry.lower() == 'n':
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断。正在退出...")
    except Exception as e:
        print(f"\n严重错误: {str(e)}")
    finally:
        print("\n程序已终止。")