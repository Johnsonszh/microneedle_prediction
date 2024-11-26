import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import itertools

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


# 定义神经网络模型
class MicroneedleQualityModel(nn.Module):
    """
    用于预测微针打印质量的前馈神经网络模型
    输入:打印参数(input_features)
    输出:各项质量评分(quality_score_components)
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # 全连接层将输入特征映射到高维空间
        self.layer1 = nn.Linear(num_inputs, 128)
        self.act1 = nn.ReLU()  # ReLU激活引入非线性
        # 第二层进一步提取特征之间的复杂关系
        self.layer2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        # 第三层逐渐降低特征维度
        self.layer3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()
        # 输出层,每个神经元对应一项质量评分
        self.output = nn.Linear(64, num_outputs)

    def forward(self, x):
        """
        定义数据在神经网络中的传播和转换过程
        """
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.output(x)
        return x


def train_model(X_train, y_train, X_val, y_val, num_epochs=500, batch_size=32):
    """
    训练微针质量预测模型
    输入:
      X_train,y_train:训练数据的输入和输出
      X_val,y_val:验证数据的输入和输出
      num_epochs:训练轮数
      batch_size:批量大小
    输出:
      训练好的模型
    """
    # 将numpy数组转换为PyTorch的Dataset
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    # 将Dataset封装成DataLoader,可按批次遍历
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化神经网络模型
    model = MicroneedleQualityModel(len(input_features), len(quality_score_components))
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters())  # Adam优化器

    best_loss = np.inf  # 初始化最优损失为无穷大
    best_epoch = 0  # 最优损失对应的轮次
    best_model_weights = None  # 最优模型参数

    for epoch in range(num_epochs):
        # 遍历训练数据的迭代器
        for inputs, targets in train_loader:
            outputs = model(inputs.float())  # 前向传播
            loss = criterion(outputs, targets.float())  # 计算损失

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

        # 训练结束后,在验证集上评估模型
        with torch.no_grad():  # 不计算梯度
            val_losses = []
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs.float())  # 前向传播
                val_loss = criterion(val_outputs, val_targets.float())  # 计算损失
                val_losses.append(val_loss.item())

        current_loss = np.mean(val_losses)  # 计算平均验证损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {current_loss:.4f}")

        if current_loss < best_loss:  # 如果当前损失是最优的
            best_loss = current_loss  # 更新最优损失
            best_epoch = epoch  # 更新最优轮次
            best_model_weights = model.state_dict()  # 保存当前的模型参数
        else:
            if epoch - best_epoch >= 20:  # 连续20轮未改善,则早停
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_weights)  # 加载之前最优的模型参数
                break

    return model


def predict_quality(model, params):
    """
    用训练好的模型预测给定参数下微针的质量
    输入:
      model:训练好的PyTorch模型
      params:包含input_features中定义的打印参数的字典
    输出:
      output_dict:预测的quality_score_components中定义的各项质量评分
      overall_score:加权求和得到的总体质量评分
    """
    input_vec = np.array([params[f] for f in input_features])  # 提取输入参数

    with torch.no_grad():  # 预测模式,不跟踪梯度
        input_tensor = torch.from_numpy(input_vec.astype(np.float32))  # 转为tensor
        output_tensor = model(input_tensor)  # 输入模型得到输出

    output_vec = output_tensor.numpy()  # 将输出tensor转为numpy数组
    output_dict = {comp: output_vec[i] for i, comp in enumerate(quality_score_components)}  # 转为字典

    overall_score = sum(output_dict[comp] * quality_score_weights[comp]
                        for comp in quality_score_components)  # 加权求总分

    return output_dict, overall_score


def grid_search(model, param_ranges, num_samples=1000):
    """
    对参数空间进行网格搜索,找到质量评分最高的参数组合
    输入:
      model:训练好的PyTorch模型
      param_ranges:dict,key为input_features中的参数名,value为其可取值列表
      num_samples:网格搜索的采样点数
    输出:
      best_params:dict,质量评分最高的参数组合
      best_score:float,best_params组合下的总体质量评分
    """
    param_combinations = list(itertools.product(*param_ranges.values()))  # 生成所有参数组合

    best_params = None
    best_score = 0

    for params in param_combinations:
        param_dict = {key: value for key, value in zip(param_ranges.keys(), params)}  # 将参数值映射回参数名
        _, overall_score = predict_quality(model, param_dict)  # 用当前参数预测质量评分

        if overall_score > best_score:  # 如果当前组合评分更高,则更新
            best_score = overall_score
            best_params = param_dict

    return best_params, best_score


if __name__ == '__main__':
    # 从CSV文件加载训练和验证数据
    train_data = np.loadtxt('.venv/train_data.csv', delimiter=',', skiprows=1)
    val_data = np.loadtxt('.venv/val_data.csv', delimiter=',', skiprows=1)

    X_train = train_data[:, :len(input_features)]  # 前len(input_features)列是输入X
    y_train = train_data[:, len(input_features):]  # 剩余列是输出y
    X_val = val_data[:, :len(input_features)]
    y_val = val_data[:, len(input_features):]

    model = train_model(X_train, y_train, X_val, y_val)  # 训练模型
    torch.save(model.state_dict(), 'microneedle_quality_model.pth')  # 保存模型权重到文件

    # 定义参数搜索空间
    param_ranges = {
        'exposure_time': [3, 3.5, 4, 4.5, 5],
        'exposure_intensity': [70, 80, 90, 100],
        'layer_height': [25, 50],
        'lifting_speed': [1.5, 2, 2.5],
        'exposure_wait': [0.2, 0.5, 0.8],
        'bottom_layers': [3, 4, 5],
        'bottom_exposure_time': [4, 5, 6],
        'print_temperature': [22, 25, 28]
    }

    # 执行网格搜索
    best_params, best_score = grid_search(model, param_ranges)
    print(f"Best parameters: {best_params}")
    print(f"Best overall quality score: {best_score:.2f}")


    def input_params():
        """
        从命令行读取用户输入的打印参数值
        返回一个字典,key为input_features中定义的参数名,value为用户输入的参数值
        """
        params = {}
        print("Please input the following parameters:")
        for feature in input_features:
            value = float(input(f"{feature}: "))  # 提示用户输入每个参数的值
            params[feature] = value  # 将参数名和值以键值对的形式存入字典
        return params

    while True:
        params = input_params()  # 获取用户输入的参数
        component_scores, overall_score = predict_quality(model, params)  # 预测质量

        print(f"\nPredicted overall quality score: {overall_score:.2f}")
        print("Component scores:")
        for component, score in component_scores.items():
            print(f"{component}: {score:.2f}")

        repeat = input("\nDo you want to predict for another set of parameters? (y/n): ")
        if repeat.lower() != 'y':  # 如果用户不再继续,则退出循环
            break