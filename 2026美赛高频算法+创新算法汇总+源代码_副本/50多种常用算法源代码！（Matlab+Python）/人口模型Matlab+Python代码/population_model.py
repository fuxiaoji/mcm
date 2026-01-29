#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人口模型案例说明：
本案例实现经典的Logistic人口增长模型，也称为S型增长模型。该模型考虑了环境承载能力，
克服了Malthus模型（指数增长）未考虑资源限制的缺陷。

模型公式：
dP/dt = r*P*(1 - P/K)

其中：
- P(t) 表示t时刻的人口数量
- r 表示固有增长率（出生率减去死亡率）
- K 表示环境承载能力（环境所能容纳的最大人口数量）

模型特点：
- 当人口数量远小于K时，增长近似指数增长
- 当人口数量接近K时，增长速度逐渐减慢
- 最终人口数量将稳定在K附近

本案例通过数值方法求解微分方程，并与美国实际人口数据进行对比验证。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 定义Logistic人口增长模型
def logistic_model(t, P, r, K):
    """
    Logistic人口增长模型微分方程
    dP/dt = r*P*(1 - P/K)
    
    参数:
    t: 时间
    P: 人口数量
    r: 固有增长率
    K: 环境承载能力
    """
    dPdt = r * P * (1 - P / K)
    return dPdt

# 2. 设置模型参数
r = 0.025       # 固有增长率（每年）
K = 300e6       # 环境承载能力（3亿人）
P0 = 3.9e6      # 初始人口（1790年美国人口）
t_span = [1790, 2050]  # 时间跨度（年）
t_eval = np.linspace(1790, 2050, 261)  # 评估时间点

# 3. 求解微分方程
solution = solve_ivp(
    logistic_model,          # 微分方程函数
    t_span,                  # 时间范围
    [P0],                    # 初始条件
    args=(r, K),             # 传递给模型的额外参数
    t_eval=t_eval,           # 评估解的时间点
    method='RK45'            # 数值方法（龙格-库塔法）
)

# 4. 美国实际人口数据（1790-2020年，单位：百万人）
years_actual = np.array([1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900,
                         1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
populations_actual = np.array([3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4, 38.6, 50.2, 62.9, 76.0,
                               92.0, 105.7, 122.8, 131.7, 150.7, 179.3, 203.2, 226.5, 248.7, 281.4, 308.7, 331.9])

# 5. 可视化结果
plt.figure(figsize=(12, 7))

# 绘制模型预测结果
plt.plot(solution.t, solution.y[0]/1e6, 'b-', linewidth=2, 
         label=f'Logistic模型预测 (r={r}, K={K/1e6}百万)')

# 绘制实际人口数据
plt.scatter(years_actual, populations_actual, color='red', s=50, 
            label='美国实际人口数据')

# 添加标签和标题
plt.xlabel('年份', fontsize=12)
plt.ylabel('人口数量（百万）', fontsize=12)
plt.title('Logistic人口增长模型与实际人口对比', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(1780, 2060)
plt.ylim(0, 350)

# 添加模型公式说明
plt.text(1800, 300, r'$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$', 
         fontsize=16, bbox=dict(facecolor='white', alpha=0.8))

plt.show()

# 6. 预测未来人口
future_years = np.array([2030, 2040, 2050])
future_indices = np.searchsorted(solution.t, future_years)
future_populations = solution.y[0][future_indices] / 1e6

print("\n未来人口预测：")
for year, pop in zip(future_years, future_populations):
    print(f"{year}年: {pop:.2f} 百万人")
    