#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据包络法(DEA)案例说明：
本案例对6家制造企业的生产效率进行评价，考虑2项投入和2项产出：
投入指标：1. 固定资产(万元)  2. 员工人数(人)
产出指标：1. 总产值(万元)    2. 净利润(万元)
通过DEA-CCR模型（投入导向）计算各企业的效率值，判断是否DEA有效，
效率值为1且松弛变量为0的企业为DEA有效。
"""

import numpy as np
from scipy.optimize import linprog

# 投入矩阵（6家企业×2项投入）
# 行：企业1-6，列：固定资产、员工人数
X = np.array([
    [500, 80],    # 企业1
    [600, 90],    # 企业2
    [400, 70],    # 企业3
    [700, 100],   # 企业4
    [300, 60],    # 企业5
    [550, 85]     # 企业6
])

# 产出矩阵（6家企业×2项产出）
# 行：企业1-6，列：总产值、净利润
Y = np.array([
    [1200, 300],  # 企业1
    [1350, 320],  # 企业2
    [1000, 280],  # 企业3
    [1500, 350],  # 企业4
    [800, 220],   # 企业5
    [1250, 310]   # 企业6
])

n, m = X.shape  # n=决策单元数, m=投入指标数
s = Y.shape[1]  # s=产出指标数

# 初始化结果变量
theta = np.zeros(n)          # 效率值
lambda_ = np.zeros((n, n))   # 权重向量
s_minus = np.zeros((n, m))   # 投入松弛变量
s_plus = np.zeros((n, s))    # 产出松弛变量

# 对每个决策单元求解DEA模型
for j in range(n):
    # 目标函数系数：min θ + ε*(sum(s⁻)+sum(s⁺))
    c = np.hstack((np.zeros(n), np.ones(m), np.ones(s)))
    
    # 等式约束矩阵
    # 投入约束: X*λ + s⁻ = θ*Xj
    A_eq = np.hstack((X.T, np.eye(m), np.zeros((m, s))))
    b_eq = X[j, :].reshape(-1, 1)
    
    # 产出约束: Y*λ - s⁺ = Yj
    A_eq = np.vstack((A_eq, np.hstack((-Y.T, np.zeros((s, m)), np.eye(s)))))
    b_eq = np.vstack((b_eq, -Y[j, :].reshape(-1, 1)))
    
    # 变量下界：λ ≥ 0, s⁻ ≥ 0, s⁺ ≥ 0
    bounds = [(0, None) for _ in range(n + m + s)]
    
    # 求解线性规划
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # 提取结果
    theta[j] = result.fun
    lambda_[j, :] = result.x[:n]
    s_minus[j, :] = result.x[n:n+m]
    s_plus[j, :] = result.x[n+m:n+m+s]

# 判断是否DEA有效（效率值=1且松弛变量全为0）
is_efficient = (theta >= 1 - 1e-6) & \
               (np.sum(s_minus, axis=1) < 1e-6) & \
               (np.sum(s_plus, axis=1) < 1e-6)

# 输出结果
print("数据包络法(DEA-CCR模型)计算结果：")
print(f"决策单元数: {n}, 投入指标数: {m}, 产出指标数: {s}")
print("\n各企业效率值及DEA有效性：")
for i in range(n):
    print(f"企业{i+1}: 效率值={theta[i]:.4f}, " + 
          ("DEA有效" if is_efficient[i] else "DEA无效"))
    if not is_efficient[i]:
        print(f"  投入松弛变量: {s_minus[i].round(4)}, 产出松弛变量: {s_plus[i].round(4)}")
