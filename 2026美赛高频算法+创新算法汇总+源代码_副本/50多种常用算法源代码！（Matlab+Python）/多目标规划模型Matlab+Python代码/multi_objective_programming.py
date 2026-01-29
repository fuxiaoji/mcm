#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多目标规划模型案例说明：
本案例使用多目标规划解决工厂生产规划问题。工厂生产两种产品A和B，
有两个目标：1) 最大化利润；2) 最小化污染物排放。
多目标规划适用于存在多个需要同时优化的目标函数的问题，通常需要找到帕累托最优解。
"""

import numpy as np
from scipy.optimize import minimize

# 定义目标函数向量
def objectives(x):
    x1, x2 = x  # x1:产品A产量, x2:产品B产量
    
    # 目标1：最大化利润（返回负值用于最小化求解）
    profit = 50*x1 + 80*x2
    profit_neg = -profit  # 转为最小化问题
    
    # 目标2：最小化污染物排放
    emission = 0.2*x1**2 + 0.5*x2**2
    
    return np.array([profit_neg, emission])

# 定义约束条件
def constraint1(x):
    # 原材料约束：3x1 + 5x2 <= 150
    return 150 - (3*x[0] + 5*x[1])

def constraint2(x):
    # 工时约束：2x1 + 3x2 <= 90
    return 90 - (2*x[0] + 3*x[1])

# 初始猜测值
x0 = [10, 10]

# 变量边界：x1 >= 0, x2 >= 0
bounds = [(0, None), (0, None)]

# 约束条件设置
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
constraints = [con1, con2]

# 使用加权求和法将多目标转化为单目标
def weighted_objective(x, weights):
    objs = objectives(x)
    return np.dot(weights, objs)

# 尝试不同权重组合，探索帕累托最优前沿
weight_combinations = [
    [0.1, 0.9],  # 更重视减排
    [0.5, 0.5],  # 均衡考虑
    [0.9, 0.1]   # 更重视利润
]

print("多目标规划生产问题结果：")
print("不同权重组合下的最优解：\n")

for i, weights in enumerate(weight_combinations):
    # 求解加权单目标问题
    solution = minimize(
        weighted_objective, 
        x0, 
        args=(weights,),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    x = solution.x
    x1, x2 = x
    
    # 计算各目标值
    objs = objectives(x)
    profit = -objs[0]  # 还原为利润正值
    emission = objs[1]
    
    # 计算约束使用情况
    material_used = 3*x1 + 5*x2
    labor_used = 2*x1 + 3*x2
    
    # 输出结果
    print(f"权重组合 {i+1}: 利润权重={weights[0]}, 减排权重={weights[1]}")
    print(f"  最优解: 产品A={x1:.2f}件, 产品B={x2:.2f}件")
    print(f"  目标值: 利润={profit:.2f}元, 污染物排放={emission:.2f}单位")
    print(f"  约束使用: 原材料={material_used:.2f}/150, 工时={labor_used:.2f}/90\n")
    