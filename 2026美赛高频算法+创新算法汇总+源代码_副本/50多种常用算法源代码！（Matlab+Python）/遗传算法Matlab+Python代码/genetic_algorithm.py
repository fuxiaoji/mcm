#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法案例说明：
本案例使用遗传算法求解Griewank函数的最小值。
Griewank函数是一个具有大量局部最优解的多峰函数，全局最小值为0，位于(0,0,...,0)。
遗传算法模拟生物进化过程，通过选择、交叉和变异操作寻找最优解，
适合求解复杂的全局优化问题。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Griewank函数：一个复杂的多峰函数
def griewank(x):
    sum_part = sum([xi**2 for xi in x]) / 4000
    prod_part = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_part - prod_part + 1

# 初始化种群
def initialize_population(pop_size, dim, bounds):
    population = []
    for _ in range(pop_size):
        individual = [np.random.uniform(low, high) for low, high in bounds]
        population.append(individual)
    return np.array(population)

# 选择操作（轮盘赌选择）
def select(population, fitness, num_parents):
    # 适应度越小越好，转换为选择概率（取倒数）
    fitness = np.max(fitness) - fitness + 1e-10  # 确保非负
    probabilities = fitness / np.sum(fitness)
    
    # 选择父代
    parents = []
    for _ in range(num_parents):
        idx = np.random.choice(len(population), p=probabilities)
        parents.append(population[idx])
    
    return np.array(parents)

# 交叉操作（单点交叉）
def crossover(parents, offspring_size):
    offspring = []
    crossover_rate = 0.8  # 交叉概率
    
    for i in range(offspring_size[0]):
        # 随机选择两个父代
        parent1_idx = i % len(parents)
        parent2_idx = (i + 1) % len(parents)
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        
        # 以一定概率进行交叉
        if np.random.rand() < crossover_rate:
            # 随机选择交叉点
            crossover_point = np.random.randint(1, len(parent1))
            # 生成子代
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring.append(child)
        else:
            # 不交叉，直接复制父代
            offspring.append(parent1.copy())
    
    return np.array(offspring)

# 变异操作
def mutate(offspring, bounds, mutation_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            # 以一定概率进行变异
            if np.random.rand() < mutation_rate:
                # 高斯变异
                offspring[i][j] += np.random.normal(0, 0.5)
                # 边界处理
                offspring[i][j] = max(bounds[j][0], min(offspring[i][j], bounds[j][1]))
    return offspring

# 遗传算法
def genetic_algorithm(objective_func, bounds, pop_size, num_generations):
    dim = len(bounds)
    
    # 初始化种群
    population = initialize_population(pop_size, dim, bounds)
    
    # 评估初始种群
    fitness = np.array([objective_func(ind) for ind in population])
    
    # 记录最优解
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_value = fitness[best_idx]
    best_history = [best_value]
    
    # 主循环
    for gen in range(num_generations):
        # 选择父代
        num_parents = pop_size // 2
        parents = select(population, fitness, num_parents)
        
        # 交叉产生子代
        offspring_size = (pop_size - num_parents, dim)
        offspring = crossover(parents, offspring_size)
        
        # 变异
        mutation_rate = 0.1  # 变异概率
        offspring = mutate(offspring, bounds, mutation_rate)
        
        # 形成新种群
        population = np.concatenate([parents, offspring])
        
        # 评估新种群
        fitness = np.array([objective_func(ind) for ind in population])
        
        # 更新最优解
        current_best_idx = np.argmin(fitness)
        current_best_value = fitness[current_best_idx]
        if current_best_value < best_value:
            best_solution = population[current_best_idx].copy()
            best_value = current_best_value
        
        # 记录历史
        best_history.append(best_value)
    
    return best_solution, best_value, best_history

# 参数设置
bounds = [(-600, 600), (-600, 600)]  # 变量范围
pop_size = 50                        # 种群大小
num_generations = 100                # 进化代数

# 运行遗传算法
best_solution, best_value, best_history = genetic_algorithm(
    griewank, bounds, pop_size, num_generations
)

# 输出结果
print("遗传算法求解Griewank函数结果：")
print(f"最优解: x = {best_solution[0]:.6f}, y = {best_solution[1]:.6f}")
print(f"最优值: f(x,y) = {best_value:.6f}")
print(f"理论最优解: (0, 0)，最优值: 0")

# 可视化结果（可选）
if True:
    # 绘制函数曲面
    fig = plt.figure(figsize=(15, 6))
    
    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([griewank([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.scatter(best_solution[0], best_solution[1], best_value, color='red', s=100, label='最优解')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Griewank函数曲面与最优解')
    ax1.legend()
    
    # 收敛曲线图
    ax2 = fig.add_subplot(122)
    ax2.plot(best_history)
    ax2.set_xlabel('进化代数')
    ax2.set_ylabel('最优值')
    ax2.set_title('收敛曲线')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    