#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
层次分析法(AHP)案例说明：
本案例用于评价3个备选方案，通过3个评价指标进行比较。
判断矩阵A中，A[i,j]表示第i个指标相对于第j个指标的重要性程度，
采用1-9标度法：1表示同等重要，3表示稍微重要，5表示明显重要，
7表示强烈重要，9表示极端重要，倒数表示相反关系。
"""

import numpy as np

# 构建判断矩阵（3个指标之间的重要性比较）
A = np.array([
    [1, 2, 3],       # 指标1与其他指标的比较：比指标2稍重要，比指标3明显重要
    [1/2, 1, 2],     # 指标2与其他指标的比较：比指标1次要，比指标3稍重要
    [1/3, 1/2, 1]    # 指标3与其他指标的比较：比指标1和2都次要
])

n = A.shape[0]  # 获取判断矩阵的阶数

# 计算最大特征值和对应的特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
max_eig = np.max(eigenvalues)  # 最大特征值
idx = np.argmax(eigenvalues)   # 最大特征值对应的索引
w = eigenvectors[:, idx].real  # 获取对应的特征向量（取实部）
w = w / np.sum(w)              # 归一化处理，得到权重向量

# 一致性检验
CI = (max_eig - n) / (n - 1)   # 计算一致性指标
# 平均随机一致性指标RI（对应1-11阶矩阵）
RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
CR = CI / RI[n] if RI[n] != 0 else 0  # 计算一致性比率

# 输出结果
print("层次分析法计算结果：")
print(f"判断矩阵阶数: {n}")
print(f"最大特征值: {max_eig:.4f}")
print(f"指标权重向量: {w.round(4)}")
print(f"一致性指标CI: {CI:.4f}")
print(f"一致性比率CR: {CR:.4f}")
print("判断矩阵" + ("通过" if CR < 0.1 else "未通过") + "一致性检验")
