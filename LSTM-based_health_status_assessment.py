import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# 数据为 "Datetime,AEP_MW" 格式
filepath = 'data/energy.csv'
data = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')

# 输出数据的分布特征
print("数据的基本统计信息：")
print(data.describe())

# 数据随时间的变化
data.plot(y='AEP_MW', figsize=(10, 6), title="Energy Data Over Time")
plt.xlabel("Datetime")
plt.ylabel("AEP_MW")
plt.show()

# 数据分布的直方图
data['AEP_MW'].plot(kind='hist', bins=50, figsize=(10, 6), title="Energy Data Distribution")
plt.xlabel("AEP_MW")
plt.show()

# 获取能量数据
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

#  LSTM 编解码器网络
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(X.shape[1], 1), return_sequences=True),
    tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 使用正常状态数据训练 LSTM 模型
model.fit(X, y, epochs=20, batch_size=64, verbose=1)

# 从模型中获取正常状态特征空间 A
encoder_output = model.predict(X)

# 计算正常状态特征空间 A 的平均距离 σ_N
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

avg_distance = compute_avg_distance(encoder_output)

# 计算实时输入数据的健康度
def calculate_health_status(current_data):
    # 将实时数据归一化
    current_data_normalized = scaler.transform(np.array(current_data).reshape(-1, 1))

    # 通过模型获取实时数据所对应特征向量
    real_time_feature = model.predict(current_data_normalized.reshape(1, -1, 1))

    # 计算实时特征与正常状态特征空间 A 的平均欧氏距离
    dist = pairwise_distances(real_time_feature, encoder_output)
    sigma_t = np.mean(dist)
    print(f"实时特征与正常状态特征空间 A 的平均欧氏距离 sigma_t: {sigma_t}")

    # σ_N 是正常状态特征空间 A 的平均距离
    sigma_N = avg_distance
    print(f"正常状态特征空间 A 的平均距离 sigma_N: {sigma_N}")

    # 健康度计算， H_t 在 0 到 100 之间
    if sigma_t > sigma_N:
        H_t = max(0, 100 - (sigma_t - sigma_N) / sigma_N * 100)
    else:
        H_t = min(100, 100 - (sigma_N - sigma_t) / sigma_N * 100)

    return round(H_t, 2)


real_time_data = [15001]  # 假设这是从实时监测中获取的数据
health_status = calculate_health_status(real_time_data)
print(f"AEP_MW: 15001对应的健康度: {health_status}")


real_time_data = [11000]
health_status = calculate_health_status(real_time_data)
print(f"AEP_MW: 11000对应的健康度: {health_status}")