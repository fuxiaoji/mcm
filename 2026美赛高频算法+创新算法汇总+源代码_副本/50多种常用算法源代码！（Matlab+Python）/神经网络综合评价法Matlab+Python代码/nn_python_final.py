#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经网络评价算法案例说明：
本案例使用神经网络对10个城市的发展水平进行评价建模，
基于5项指标（GDP、人口、就业率、教育投入、医疗资源）训练模型，
然后用训练好的模型对5个新城市进行发展水平评分和排序。
神经网络通过学习已知评分的城市数据，能够捕捉指标间的非线性关系，
提供更精准的综合评价结果。
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 1. 准备训练数据
# 训练数据：10个城市，5项指标（GDP(亿元)、人口(万)、就业率(%)、教育投入(亿元)、医疗资源(床位数/万人)
train_data = np.array([
    [3200, 280, 92.5, 85, 6.2],
    [2800, 220, 91.3, 78, 5.8],
    [4100, 350, 93.1, 92, 6.5],
    [1800, 150, 89.7, 65, 4.9],
    [5200, 420, 94.2, 105, 7.1],
    [2500, 200, 90.5, 72, 5.5],
    [3800, 320, 92.8, 88, 6.3],
    [1500, 130, 88.9, 60, 4.7],
    [4500, 380, 93.5, 96, 6.8],
    [2200, 180, 89.9, 68, 5.2]
])

# 训练标签：专家对10个城市的发展水平评分（0-100分）
train_labels = np.array([78, 72, 85, 65, 92, 69, 82, 62, 88, 67])

# 2. 准备测试数据（5个新城市）
test_data = np.array([
    [3600, 300, 92.2, 86, 6.4],   # 城市A
    [2900, 230, 91.5, 79, 5.9],   # 城市B
    [4800, 400, 93.8, 98, 6.9],   # 城市C
    [2100, 170, 89.8, 67, 5.1],   # 城市D
    [3300, 290, 92.0, 84, 6.3]    # 城市E
])

# 3. 数据标准化（均值为0，标准差为1）
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)  # 拟合训练数据并标准化
test_data_scaled = scaler.transform(test_data)        # 使用相同的缩放参数标准化测试数据

# 4. 创建并训练神经网络模型
model = MLPRegressor(
    hidden_layer_sizes=(10,),    # 1个隐藏层，10个神经元
    activation='relu',           # 激活函数
    solver='adam',               # 优化器
    max_iter=1000,               # 最大迭代次数
    learning_rate_init=0.001,    # 初始学习率
    random_state=42,             # 随机种子，保证结果可复现
    verbose=False                # 不输出训练过程
)

# 训练模型
model.fit(train_data_scaled, train_labels)

# 5. 预测测试数据
test_scores = model.predict(test_data_scaled)

# 6. 排序（从高到低）
sorted_indices = np.argsort(-test_scores)

# 输出结果
print("神经网络评价算法结果：")
print(f"模型在训练数据上的R²得分: {model.score(train_data_scaled, train_labels):.4f}")
print("\n测试城市的发展水平评分及排名：")
for i, idx in enumerate(sorted_indices):
    city_name = chr(65 + idx)  # A, B, C, D, E
    print(f"第{i+1}名: 城市{city_name}, 评分: {test_scores[idx]:.2f}")
