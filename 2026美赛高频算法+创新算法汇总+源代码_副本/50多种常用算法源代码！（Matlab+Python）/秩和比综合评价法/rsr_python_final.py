#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
秩和比综合评价法案例说明：
本案例对8所高校的综合实力进行评价，考虑3项指标：
1. 科研经费(万元)  2. 师资力量(教授人数)  3. 就业率(%)
所有指标均为效益型指标（数值越大越好），通过秩和比法计算RSR值，
并进行分档（1-4档，1档最优）和排序，评价各高校的综合实力。
"""

import numpy as np

# 原始数据矩阵（8个样本×3个指标）
# 行：高校1-8，列：科研经费、师资力量、就业率
data = np.array([
    [8500, 90, 78],   # 高校1
    [9200, 85, 90],   # 高校2
    [7800, 92, 86],   # 高校3
    [8800, 88, 94],   # 高校4
    [9000, 82, 89],   # 高校5
    [8200, 87, 91],   # 高校6
    [8600, 91, 83],   # 高校7
    [8900, 84, 87]    # 高校8
])

# 各指标权重
weights = np.array([0.3, 0.4, 0.3])

# 指标类型（True表示效益型，False表示成本型）
is_benefit = np.array([True, True, True])

m, n = data.shape  # m=样本数, n=指标数

# 1. 编秩（对每个指标进行排序并赋值秩次）
R = np.zeros((m, n))  # 秩矩阵
for j in range(n):
    # 按降序排序并获取索引（从大到小）
    idx = np.argsort(-data[:, j])
    r = np.zeros(m)
    for i in range(m):
        r[idx[i]] = i + 1  # 秩从1到m（最大值秩为1）
    
    # 对于成本型指标，反转秩次（最小值秩为1）
    if not is_benefit[j]:
        r = m + 1 - r
    
    R[:, j] = r

# 2. 计算加权秩和比RSR
rsr = np.dot(R, weights) / m  # 加权求和后除以样本数

# 3. 排序（从优到劣）
sorted_indices = np.argsort(-rsr)
rank = np.zeros(m, dtype=int)
for i in range(m):
    rank[sorted_indices[i]] = i + 1  # 排名从1开始

# 4. 分档（分为4档，1档最优）
grade = np.zeros(m, dtype=int)
quarter = np.ceil(m / 4)  # 每档的大致数量
grade[sorted_indices[:int(quarter)]] = 1
grade[sorted_indices[int(quarter):int(2*quarter)]] = 2
grade[sorted_indices[int(2*quarter):int(3*quarter)]] = 3
grade[sorted_indices[int(3*quarter):]] = 4

# 输出结果
print("秩和比综合评价法计算结果：")
print(f"样本数: {m}, 指标数: {n}")
print(f"各指标权重: {weights.round(4)}")
print("\n各高校秩和比(RSR)值、排名及分档：")
for i in range(m):
    print(f"高校{i+1}: RSR值={rsr[i]:.4f}, 排名={rank[i]}, 档级={grade[i]}")
