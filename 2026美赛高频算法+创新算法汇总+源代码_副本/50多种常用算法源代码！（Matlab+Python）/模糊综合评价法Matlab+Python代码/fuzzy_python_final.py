#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模糊综合评价法案例说明：
本案例对5名员工的工作表现进行评价，考虑3项指标：
1. 工作效率(越高越好)  2. 团队协作(越高越好)  3. 创新能力(越高越好)
评价等级分为3级：优、中、差，通过模糊综合评价法计算各员工的隶属度，
并给出综合得分和排名。
"""

import numpy as np

# 原始数据矩阵（5个样本×3个指标）
# 行：员工1-5，列：工作效率、团队协作、创新能力
data = np.array([
    [85, 90, 88],   # 员工1
    [92, 85, 90],   # 员工2
    [78, 92, 86],   # 员工3
    [88, 88, 94],   # 员工4
    [90, 82, 89]    # 员工5
])

# 各指标权重
weights = np.array([0.3, 0.4, 0.3])

# 评价等级标准（3个等级：优、中、差）
# 每行对应一个指标的三个等级标准
criteria = np.array([
    [90, 80, 70],  # 工作效率的优、中、差标准
    [90, 80, 70],  # 团队协作的优、中、差标准
    [90, 80, 70]   # 创新能力的优、中、差标准
])

m, n = data.shape  # m=样本数, n=指标数
k = criteria.shape[1]  # k=评价等级数

# 初始化隶属度矩阵（样本数×等级数×指标数）
membership = np.zeros((m, k, n))

# 计算隶属度
for i in range(m):        # 遍历每个样本
    for j in range(n):    # 遍历每个指标
        x = data[i, j]    # 当前样本的当前指标值
        s = criteria[j, :]# 当前指标的评价标准
        
        # 计算第一个等级(优)的隶属度
        if x >= s[0]:
            membership[i, 0, j] = 1
        elif x < s[1]:
            membership[i, 0, j] = 0
        else:
            # 线性隶属函数
            membership[i, 0, j] = (x - s[1]) / (s[0] - s[1])
        
        # 计算中间等级(中)的隶属度
        if k > 2:
            if x >= s[0] or x < s[2]:
                membership[i, 1, j] = 0
            elif x >= s[1]:
                membership[i, 1, j] = (s[0] - x) / (s[0] - s[1])
            else:
                membership[i, 1, j] = (x - s[2]) / (s[1] - s[2])
        
        # 计算最后一个等级(差)的隶属度
        if x <= s[-1]:
            membership[i, -1, j] = 1
        elif x > s[-2]:
            membership[i, -1, j] = 0
        else:
            membership[i, -1, j] = (s[-2] - x) / (s[-2] - s[-1])

# 计算综合评价矩阵（加权求和）
evaluation = np.zeros((m, k))
for i in range(m):
    for l in range(k):
        # 对每个等级，计算加权后的隶属度
        evaluation[i, l] = np.sum(weights * membership[i, l, :])

# 计算综合得分（等级赋值：优=3, 中=2, 差=1）
level_scores = np.arange(k, 0, -1)  # 从高到低的得分
final_scores = evaluation @ level_scores  # 矩阵乘法计算得分

# 排序（从优到劣）
sorted_indices = np.argsort(-final_scores)

# 输出结果
print("模糊综合评价法计算结果：")
print("评价等级标准（优、中、差）：")
for j in range(n):
    print(f"指标{j+1}标准: {criteria[j, :]}")
print("\n综合评价矩阵（每行一个样本，每列一个等级）：")
print(evaluation.round(4))
print("\n各样本综合得分及排名：")
for i, idx in enumerate(sorted_indices):
    print(f"第{i+1}名: 员工{idx+1}, 得分: {final_scores[idx]:.4f}")
