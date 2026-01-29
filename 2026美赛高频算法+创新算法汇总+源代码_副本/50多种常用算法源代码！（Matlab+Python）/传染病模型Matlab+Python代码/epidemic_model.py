#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传染病模型案例说明：
本案例实现经典的SIR传染病模型，该模型将人群分为三类：
- S (Susceptible)：易感人群，可能被感染的健康人群
- I (Infected)：感染人群，已经感染并具有传染性的人群
- R (Recovered)：康复人群，已经康复并获得免疫力的人群

模型公式：
dS/dt = -β*S*I/N
dI/dt = β*S*I/N - γ*I
dR/dt = γ*I

其中：
- N = S + I + R 表示总人口数（假设恒定）
- β 表示感染率（每个感染者单位时间内传染的易感者数量）
- γ 表示恢复率（单位时间内从感染者中康复的比例）
- R0 = β/γ 表示基本再生数，衡量病毒传播能力

模型特点：
- 描述了传染病在人群中的传播、发展和消退过程
- R0 > 1 时，传染病会流行；R0 < 1 时，传染病会逐渐消失
- 感染人群通常会先增加到一个峰值，然后逐渐减少

本案例模拟不同参数下的疫情发展曲线，并分析基本再生数R0对疫情的影响。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 定义SIR传染病模型
def sir_model(t, state, beta, gamma, N):
    """
    SIR传染病模型微分方程组
    
    参数:
    t: 时间
    state: 状态变量 [S, I, R]
    beta: 感染率
    gamma: 恢复率
    N: 总人口数
    """
    S, I, R = state
    dSdt = -beta * S * I / N    # 易感人群变化率
    dIdt = beta * S * I / N - gamma * I  # 感染人群变化率
    dRdt = gamma * I            # 康复人群变化率
    return [dSdt, dIdt, dRdt]

# 2. 设置模型参数和初始条件
N = 100000               # 总人口数
I0 = 10                  # 初始感染人数
R0_initial = 0           # 初始康复人数
S0 = N - I0 - R0_initial # 初始易感人数

# 三种不同传播能力的场景
scenarios = [
    {"beta": 0.3, "gamma": 0.1, "label": "R0=3.0 (高传播性)"},  # R0=β/γ=3.0
    {"beta": 0.15, "gamma": 0.1, "label": "R0=1.5 (中传播性)"}, # R0=1.5
    {"beta": 0.08, "gamma": 0.1, "label": "R0=0.8 (低传播性)"}  # R0=0.8
]

t_span = [0, 100]  # 时间跨度（天）
t_eval = np.linspace(0, 100, 1000)  # 评估时间点

# 3. 求解微分方程（三种场景）
solutions = []
for scenario in scenarios:
    beta = scenario["beta"]
    gamma = scenario["gamma"]
    
    solution = solve_ivp(
        sir_model,                # 微分方程函数
        t_span,                    # 时间范围
        [S0, I0, R0_initial],      # 初始条件
        args=(beta, gamma, N),     # 传递给模型的额外参数
        t_eval=t_eval,             # 评估解的时间点
        method='RK45'              # 数值方法
    )
    solutions.append(solution)

# 4. 分析关键指标
def analyze_scenario(solution, N, label):
    """分析疫情的关键指标"""
    S, I, R = solution.y
    # 感染人数峰值及出现时间
    peak_idx = np.argmax(I)
    peak_I = I[peak_idx]
    peak_time = solution.t[peak_idx]
    # 最终感染比例
    final_infected_ratio = (N - S[-1]) / N * 100  # 初始易感者中最终被感染的比例
    return {
        "label": label,
        "peak_time": peak_time,
        "peak_I": peak_I,
        "final_infected_ratio": final_infected_ratio
    }

# 分析所有场景
analysis_results = [
    analyze_scenario(solutions[i], N, scenarios[i]["label"])
    for i in range(len(scenarios))
]

# 5. 可视化结果
plt.figure(figsize=(14, 10))

# 绘制各人群随时间变化曲线（三种场景对比）
plt.subplot(2, 1, 1)
for i in range(len(scenarios)):
    solution = solutions[i]
    S, I, R = solution.y
    plt.plot(solution.t, S, 'b-', linewidth=1.5, alpha=0.7)
    plt.plot(solution.t, I, 'r-', linewidth=2, label=scenarios[i]["label"])
    plt.plot(solution.t, R, 'g-', linewidth=1.5, alpha=0.7)

plt.xlabel('时间（天）', fontsize=12)
plt.ylabel('人数', fontsize=12)
plt.title('不同传播能力下的SIR模型曲线', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.text(5, N*0.9, 'S: 易感人群 (蓝色)\nI: 感染人群 (红色)\nR: 康复人群 (绿色)', 
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# 单独绘制感染人群曲线，突出峰值
plt.subplot(2, 1, 2)
for i in range(len(scenarios)):
    solution = solutions[i]
    I = solution.y[1]
    peak_time = analysis_results[i]["peak_time"]
    peak_I = analysis_results[i]["peak_I"]
    
    plt.plot(solution.t, I, linewidth=2, label=scenarios[i]["label"])
    # 标记峰值点
    plt.scatter(peak_time, peak_I, s=50, zorder=5)
    plt.text(peak_time+1, peak_I, f'峰值: {peak_I:.0f}人\n时间: {peak_time:.1f}天', 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('时间（天）', fontsize=12)
plt.ylabel('感染人数', fontsize=12)
plt.title('不同传播能力下的感染人群变化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

# 6. 输出分析结果
print("疫情分析结果：")
print(f"总人口数: {N}人, 初始感染人数: {I0}人")
print("-" * 70)
print(f"{'场景':<20} {'感染峰值时间(天)':<20} {'感染峰值人数':<20} {'最终感染比例(%)':<20}")
print("-" * 70)
for result in analysis_results:
    print(f"{result['label']:<20} {result['peak_time']:<20.1f} {result['peak_I']:<20.0f} {result['final_infected_ratio']:<20.1f}")
    