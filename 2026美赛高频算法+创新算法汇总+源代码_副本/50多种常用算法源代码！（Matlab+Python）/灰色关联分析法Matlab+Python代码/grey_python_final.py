#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
灰色关联分析法案例说明：
本案例对5个地区的经济发展水平进行评价，考虑4项指标：
1. GDP增长率(%)  2. 人均收入(万元)  3. 就业率(%)  4. 财政收入(亿元)
通过灰色关联分析，将各地区与理想参考序列（各项指标均为最优值）进行比较，
计算关联度并排序，判断各地区经济发展水平的优劣。
"""

import numpy as np

# 原始数据矩阵（5个地区×4个指标）
# 行：地区1-5，列：GDP增长率、人均收入、就业率、财政收入
data = np.array([
    [1.2, 3.4, 2.5, 5.6],   # 地区1
    [2.1, 4.2, 3.1, 6.2],   # 地区2
    [1.5, 2.8, 2.9, 4.8],   # 地区3
    [3.0, 3.9, 4.0, 5.9],   # 地区4
    [2.4, 3.5, 3.6, 6.0]    # 地区5
])

# 参考序列（理想方案）- 每个指标的最优值（此处取各指标最大值）
reference = np.array([3.0, 4.2, 4.0, 6.2])

m, n = data.shape  # m=样本数, n=指标数

# 数据归一化（区间化法）
# 合并数据与参考序列以便统一归一化
combined = np.vstack((data, reference))
max_val = np.max(combined, axis=0)  # 各指标最大值
min_val = np.min(combined, axis=0)  # 各指标最小值

# 初始化归一化后的数据
data_norm = np.zeros((m, n))
ref_norm = np.zeros(n)

for j in range(n):
    # 防止分母为0
    if max_val[j] != min_val[j]:
        # 归一化到[0,1]区间
        data_norm[:, j] = (data[:, j] - min_val[j]) / (max_val[j] - min_val[j])
        ref_norm[j] = (reference[j] - min_val[j]) / (max_val[j] - min_val[j])
    else:
        data_norm[:, j] = np.zeros(m)
        ref_norm[j] = 0

# 计算绝对差
delta = np.abs(data_norm - ref_norm)
max_delta = np.max(delta)  # 最大绝对差
min_delta = np.min(delta)  # 最小绝对差

# 计算关联系数（分辨系数rho通常取0.5）
rho = 0.5
gamma = (min_delta + rho * max_delta) / (delta + rho * max_delta)

# 计算关联度（各指标关联系数的平均值）
r = np.mean(gamma, axis=1)

# 排序（从大到小）
rank = np.argsort(-r) + 1  # +1使排名从1开始

# 输出结果
print("灰色关联分析法计算结果：")
print(f"参考序列（理想方案）: {reference}")
print(f"各地区与理想方案的关联度: {r.round(4)}")
print("各地区排名（从优到劣）:")
for i in range(m):
    print(f"第{rank[i]}名: 地区{np.where(rank == i+1)[0][0]+1}, 关联度: {r[np.where(rank == i+1)[0][0]]:.4f}")
