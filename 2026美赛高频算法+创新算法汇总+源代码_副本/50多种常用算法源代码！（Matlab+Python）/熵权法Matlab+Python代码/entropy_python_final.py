#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
熵权法案例说明：
本案例对5家企业的3项经济效益指标进行评价，指标分别为：
1. 净利润(万元)  2. 资产负债率(%)  3. 销售收入增长率(%)
所有指标均为效益型指标（数值越大越好），通过熵权法计算各指标的客观权重，
并对5家企业进行综合评分排序。
"""

import numpy as np

# 原始数据矩阵（5个样本×3个指标）
# 行：5家企业，列：净利润、资产负债率、销售收入增长率
data = np.array([
    [89, 90, 92],   # 企业1
    [92, 88, 90],   # 企业2
    [78, 95, 86],   # 企业3
    [90, 85, 94],   # 企业4
    [85, 92, 88]    # 企业5
])

m, n = data.shape  # m=样本数, n=指标数

# 数据归一化（正向化处理）
data_norm = np.zeros((m, n))
for j in range(n):
    max_val = np.max(data[:, j])  # 第j个指标的最大值
    min_val = np.min(data[:, j])  # 第j个指标的最小值
    # 防止分母为0
    if max_val != min_val:
        # 归一化到[0,1]区间
        data_norm[:, j] = (data[:, j] - min_val) / (max_val - min_val)
    else:
        data_norm[:, j] = np.zeros(m)  # 若指标值相同，归一化为0

# 计算第j项指标下第i个样本的比重p_ij
p = np.zeros((m, n))
for j in range(n):
    sum_col = np.sum(data_norm[:, j])  # 第j个指标归一化后总和
    if sum_col != 0:
        p[:, j] = data_norm[:, j] / sum_col  # 计算比重
    else:
        p[:, j] = np.ones(m) / m  # 若总和为0，平均分配比重

# 计算熵值e_j
e = np.zeros(n)
# 计算常数k（1/ln(m)）
if m > 1:
    k = 1 / np.log(m)
else:
    k = 0
for j in range(n):
    # 计算每个指标的熵值，添加极小值防止log(0)
    e[j] = -k * np.sum(p[:, j] * np.log(p[:, j] + np.finfo(float).eps))

# 计算信息熵冗余度d_j和权重
d = 1 - e               # 冗余度 = 1 - 熵值
weights = d / np.sum(d) # 权重归一化

# 计算各样本的综合得分
scores = np.dot(data_norm, weights)

# 输出结果
print("熵权法计算结果：")
print(f"样本数: {m}, 指标数: {n}")
print(f"各指标权重: {weights.round(4)}")
print("各样本综合得分及排名：")
# 排序并输出
sorted_indices = np.argsort(-scores)
for i, idx in enumerate(sorted_indices):
    print(f"第{i+1}名: 样本{idx+1}, 得分: {scores[idx]:.4f}")
