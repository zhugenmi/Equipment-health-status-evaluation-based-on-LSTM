import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

filepath = 'data/energy.csv'
data = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')

print("数据的基本统计信息：")
print(data.describe())

# 数据随时间的变化
data.plot(y='AEP_MW', figsize=(10, 6), title="Energy Data Over Time")
plt.xlabel("Datetime")
plt.ylabel("AEP_MW")
# plt.show()

# 数据分布的直方图
data['AEP_MW'].plot(kind='hist', bins=50, figsize=(10, 6), title="Energy Data Distribution")
plt.xlabel("AEP_MW")
# plt.show()

energy_data = data['AEP_MW'].values

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
energy_data_normalized = scaler.fit_transform(energy_data.reshape(-1, 1))

# 准备 LSTM 模型所需的时间序列数据集
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 10  # 设置时间步长
X, y = create_dataset(energy_data_normalized, time_steps)

# 将 X 形状转换为 LSTM 输入需要的 [样本数，时间步长，特征数]
X = X.reshape(X.shape[0], X.shape[1], 1)

# LSTM 编解码器网络
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), x.size(1), hidden.size(2)).to(x.device)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        output = self.fc(decoder_output[:, -1, :])
        return output

# 初始化模型
input_size = 1
hidden_size = 64
num_layers = 2
model = LSTMEncoderDecoder(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 转换数据为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 训练模型
epochs = 5
batch_size = 64
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_tensor), batch_size):
        X_batch = X_tensor[i:i + batch_size]
        y_batch = y_tensor[i:i + batch_size]

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (len(X_tensor) // batch_size):.4f}")

# 从模型中获取正常状态特征空间 A
model.eval()
with torch.no_grad():
    encoder_outputs = []
    for i in range(0, len(X_tensor), batch_size):
        X_batch = X_tensor[i:i + batch_size]
        encoded, _ = model.encoder(X_batch)
        encoder_outputs.append(encoded[:, -1, :])
    feature_space = torch.cat(encoder_outputs, dim=0).numpy()

# 计算正常状态特征空间 A 的平均距离 sigma_N
def compute_avg_distance(feature_space):
    batch_size = 1000
    num_samples = feature_space.shape[0]
    dist = 0
    count = 0
    for i in range(0, num_samples, batch_size):
        batch = feature_space[i:i + batch_size]
        dist += np.sum(pairwise_distances(batch, feature_space))
        count += batch.shape[0]
    return dist / (count * feature_space.shape[0])

avg_distance = compute_avg_distance(feature_space)

# 计算实时输入数据的健康度
def calculate_health_status(current_data):
    # 将实时数据归一化
    current_data_normalized = scaler.transform(np.array(current_data).reshape(-1, 1))
    current_data_tensor = torch.tensor(current_data_normalized, dtype=torch.float32).reshape(1, -1, 1)

    # 通过模型获取实时数据所对应特征向量
    with torch.no_grad():
        real_time_feature, _ = model.encoder(current_data_tensor)
    real_time_feature = real_time_feature[:, -1, :].numpy()

    # 计算实时特征与正常状态特征空间 A 的平均欧氏距离
    dist = pairwise_distances(real_time_feature, feature_space)
    sigma_t = np.mean(dist)
    print(f"实时特征与正常状态特征空间 A 的平均欧氏距离 sigma_t: {sigma_t}")

    # sigma_N 是正常状态特征空间 A 的平均距离
    sigma_N = avg_distance
    print(f"正常状态特征空间 A 的平均距离 sigma_N: {sigma_N}")

    # 健康度计算， H_t 在 0 到 100 之间
    if sigma_t > sigma_N:
        H_t = sigma_N  / sigma_t * 100
    else:
        H_t = 100

    return round(H_t, 2)

real_time_data = [15001]  # 假设这是从实时监测中获取的数据
health_status = calculate_health_status(real_time_data)
print(f"AEP_MW: 15001对应的健康度: {health_status}")

real_time_data = [11000]
health_status = calculate_health_status(real_time_data)
print(f"AEP_MW: 11000对应的健康度: {health_status}")
