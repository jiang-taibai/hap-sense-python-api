import datetime
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from tools.从数据库获取数据 import get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, predicted_days=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predicted_days = predicted_days
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * predicted_days)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 初始化隐藏状态
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 初始化细胞状态
        out, _ = self.lstm(x.to(device), (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        out = out.view(-1, self.predicted_days, out.shape[-1] // self.predicted_days)
        return out


def date_str_to_array(date_str):
    """
    将日期字符串转换为数组
    :param date_str: 日期字符串，格式为 'YYYY-MM-DD'
    :return: [year, month, day]
    """
    date = date_str.split('-')
    return [int(date[0]), int(date[1]), int(date[2])]


def build_dataset(data, previous_days=10, predicted_days=1):
    """
    预处理数据
    :param data: [[date, population, households, planting_households], ...]
    :param previous_days: 用之前的多少天的数据来预测
    :param predicted_days: 预测出多少天的数据
    :return:
    """
    # 转换成 [[year, month, day, population, households, planting_households], ...] 的形式
    data = [d.to_array() for d in data]
    data = [date_str_to_array(daily_data[0]) + daily_data[1:] for daily_data in data]

    data = np.array(data)

    # 构建输入和输出
    input_ground_truth, output_ground_truth = [], []
    for i in range(len(data) - previous_days - predicted_days):
        input_ground_truth.append(data[i:(i + previous_days)])
        output_ground_truth.append(data[(i + previous_days):(i + previous_days + predicted_days), 3:])

    # 转换为 PyTorch 的张量
    input_ground_truth = np.array(input_ground_truth).astype(np.float32)
    output_ground_truth = np.array(output_ground_truth).astype(np.float32)

    # 划分训练集和测试集
    split = int(len(input_ground_truth) * 0.8)
    input_train_set, input_test_set = input_ground_truth[:split], input_ground_truth[split:]
    output_train_set, output_test_set = output_ground_truth[:split], output_ground_truth[split:]

    return input_train_set, input_test_set, output_train_set, output_test_set


def normalization(input_set, output_set):
    input_set_2d = input_set.reshape(-1, input_set.shape[-1])
    output_set_2d = output_set.reshape(-1, output_set.shape[-1])
    scaler_date = MinMaxScaler()
    scaler_statistics = MinMaxScaler()
    scaler_date.fit(input_set_2d[:, :3])
    scaler_statistics.fit(input_set_2d[:, 3:])
    input_set_2d[:, :3] = scaler_date.transform(input_set_2d[:, :3])
    input_set_2d[:, 3:] = scaler_statistics.transform(input_set_2d[:, 3:])
    output_set_2d = scaler_statistics.transform(output_set_2d)
    input_set = input_set_2d.reshape(input_set.shape)
    output_set = output_set_2d.reshape(output_set.shape)
    return input_set, output_set, scaler_date, scaler_statistics


def de_normalization(data, scaler_date, scaler_statistics):
    data_2d = data.reshape(-1, data.shape[-1])
    if data_2d.shape[1] == 3:
        data_2d = scaler_statistics.inverse_transform(data_2d)
        return data_2d.reshape(data.shape)
    else:
        data_2d[:, :3] = scaler_date.inverse_transform(data_2d[:, :3])
        data_2d[:, 3:] = scaler_statistics.inverse_transform(data_2d[:, 3:])
        return data_2d.reshape(data.shape)


def train(input_train_set, input_test_set, output_train_set, output_test_set,
          save_path='model.pth',
          hidden_size=50, num_layers=1, num_epochs=50, batch_size=32, predicted_days=1):
    (input_train_set, output_train_set,
     train_set_date_scaler, train_set_statistics_scaler) = normalization(input_train_set, output_train_set)
    (input_test_set, output_test_set,
     test_set_date_scaler, test_set_statistics_scaler) = normalization(input_test_set, output_test_set)

    input_train_set = torch.tensor(input_train_set, dtype=torch.float32)
    output_train_set = torch.tensor(output_train_set, dtype=torch.float32)
    input_test_set = torch.tensor(input_test_set, dtype=torch.float32)
    output_test_set = torch.tensor(output_test_set, dtype=torch.float32)

    input_size = input_train_set.shape[2]
    output_size = output_train_set.shape[2]

    model = LSTMModel(input_size, hidden_size, output_size, num_layers, predicted_days).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        loss = None
        for i in range(0, len(input_train_set), batch_size):
            input_batch = input_train_set[i:i + batch_size].to(device)
            output_batch = output_train_set[i:i + batch_size].to(device)
            outputs = model(input_batch)    # outputs.shape=torch.Size([32, 10, 3])
            loss = criterion(outputs, output_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), save_path)

    # 测试
    model.eval()
    with torch.no_grad():
        predictions = model(input_test_set)

    # 打印部分预测值和真实值对比
    print("Predictions:", de_normalization(predictions[:5].cpu().numpy(),
                                           test_set_date_scaler,
                                           test_set_statistics_scaler).tolist())
    print("Actual:", de_normalization(output_test_set[:5].numpy(),
                                      test_set_date_scaler,
                                      test_set_statistics_scaler).tolist())
    return model


def predict(data, model_path='model.pth', input_size=10, hidden_size=50, output_size=1, num_layers=1):
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 归一化数据
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # 转换为 PyTorch 的张量
    input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # 预测
    with torch.no_grad():
        prediction = model(input_data)

    # 反归一化
    prediction = scaler.inverse_transform(prediction.numpy())
    return prediction


def main():
    data_filename = 'dataset/data.pkl'
    dataset_configs = {
        'previous_days': 100,
        'predicted_days': 10,
    }
    train_configs = {
        'save_path': f'weight/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}-test.pth',
        'hidden_size': 50 * 10,
        'num_layers': 1,
        'num_epochs': 100,
        'batch_size': 32,
        'predicted_days': 10,
    }

    if not os.path.exists(data_filename):
        with open('dataset/data.pkl', 'wb') as f:
            pickle.dump(get_data(), f)
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    (input_train_set, input_test_set,
     output_train_set, output_test_set) = build_dataset(data, **dataset_configs)
    model = train(input_train_set, input_test_set, output_train_set, output_test_set, **train_configs)


if __name__ == '__main__':
    main()
