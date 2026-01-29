#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粒子群算法案例说明：
本案例使用粒子群优化算法(PSO)求解Rastrigin函数的最小值。
Rastrigin函数是一个典型的非线性多峰函数，具有大量局部最优解，
常用于测试优化算法的性能。粒子群算法模拟鸟群觅食行为，
通过群体中个体间的协作和信息共享寻找最优解。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rastrigin函数：一个复杂的多峰函数，最小值在(0,0)处，值为0
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# 粒子群算法
def particle_swarm_optimization(objective_func, dim, bounds, num_particles, max_iter):
    # 初始化粒子群参数
    w = 0.5        # 惯性权重
    c1 = 1         # 认知系数
    c2 = 2         # 社会系数
    
    # 初始化粒子位置和速度
    particles = np.random.rand(num_particles, dim)  # 随机位置 [0,1)
    # 将位置映射到给定范围
    for i in range(dim):
        particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    
    velocities = np.random.randn(num_particles, dim) * 0.1  # 初始速度
    
    # 初始化个体最优和全局最优
    pbest_pos = particles.copy()  # 个体最优位置
    pbest_val = np.array([objective_func(p) for p in particles])  # 个体最优值
    gbest_idx = np.argmin(pbest_val)  # 全局最优索引
    gbest_pos = pbest_pos[gbest_idx].copy()  # 全局最优位置
    gbest_val = pbest_val[gbest_idx]  # 全局最优值
    
    # 记录全局最优值的变化
    gbest_history = [gbest_val]
    
    # 主循环
    for _ in range(max_iter):
        # 更新速度和位置
        for i in range(num_particles):
            # 计算新速度
            r1 = np.random.rand(dim)  # 随机因子1
            r2 = np.random.rand(dim)  # 随机因子2
            cognitive = c1 * r1 * (pbest_pos[i] - particles[i])  # 认知部分
            social = c2 * r2 * (gbest_pos - particles[i])        # 社会部分
            velocities[i] = w * velocities[i] + cognitive + social  # 新速度
            
            # 更新位置
            particles[i] += velocities[i]
            
            # 边界处理
            for j in range(dim):
                if particles[i, j] < bounds[j][0]:
                    particles[i, j] = bounds[j][0]
                    velocities[i, j] = 0  # 碰到边界速度归零
                elif particles[i, j] > bounds[j][1]:
                    particles[i, j] = bounds[j][1]
                    velocities[i, j] = 0  # 碰到边界速度归零
        
        # 评估当前位置
        current_val = np.array([objective_func(p) for p in particles])
        
        # 更新个体最优
        improved = current_val < pbest_val
        pbest_pos[improved] = particles[improved].copy()
        pbest_val[improved] = current_val[improved]
        
        # 更新全局最优
        current_gbest_idx = np.argmin(pbest_val)
        if pbest_val[current_gbest_idx] < gbest_val:
            gbest_pos = pbest_pos[current_gbest_idx].copy()
            gbest_val = pbest_val[current_gbest_idx]
        
        # 记录历史
        gbest_history.append(gbest_val)
    
    return gbest_pos, gbest_val, gbest_history

# 参数设置
dim = 2  # 问题维度
bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # 变量范围
num_particles = 30  # 粒子数量
max_iter = 100  # 最大迭代次数

# 运行粒子群算法
gbest_pos, gbest_val, gbest_history = particle_swarm_optimization(
    rastrigin, dim, bounds, num_particles, max_iter
)

# 输出结果
print("粒子群算法求解Rastrigin函数结果：")
print(f"最优解: x = {gbest_pos[0]:.6f}, y = {gbest_pos[1]:.6f}")
print(f"最优值: f(x,y) = {gbest_val:.6f}")

# 可视化结果（可选）
if True:
    # 绘制函数曲面
    fig = plt.figure(figsize=(15, 6))
    
    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rastrigin([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.scatter(gbest_pos[0], gbest_pos[1], gbest_val, color='red', s=100, label='最优解')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Rastrigin函数曲面与最优解')
    ax1.legend()
    
    # 收敛曲线图
    ax2 = fig.add_subplot(122)
    ax2.plot(gbest_history)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('全局最优值')
    ax2.set_title('收敛曲线')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    