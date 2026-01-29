#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高斯混合聚类模型（GMM）案例说明：
高斯混合模型假设数据是由多个高斯分布的混合生成的，通过期望最大化（EM）算法估计每个高斯分布的参数。
与K-means不同，GMM是软聚类方法，每个样本属于每个簇的概率是连续的。

算法特点：
1. 假设数据由k个高斯分布混合而成
2. 使用EM算法估计每个高斯分布的均值、协方差和权重
3. 可以处理非球形簇和重叠簇
4. 输出每个样本属于各个簇的概率

本案例使用生成的二维模拟数据（3个不同的高斯分布），通过GMM进行聚类，
并可视化聚类结果和每个高斯分布的轮廓。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 生成模拟数据（3个不同的高斯分布）
np.random.seed(42)  # 设置随机种子，确保结果可复现

# 定义3个高斯分布的参数
means = [[0, 5], [5, 0], [10, 5]]  # 均值
covariances = [
    [[2, 1], [1, 2]],    # 协方差矩阵1
    [[3, -1], [-1, 2]],  # 协方差矩阵2
    [[1, 0], [0, 3]]     # 协方差矩阵3
]
weights = [0.3, 0.5, 0.2]  # 权重
n_samples = 500  # 总样本数

# 生成数据
X = np.zeros((n_samples, 2))
y_true = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    # 随机选择一个高斯分布
    cluster = np.random.choice(3, p=weights)
    y_true[i] = cluster
    # 从选中的高斯分布生成样本
    X[i] = np.random.multivariate_normal(means[cluster], covariances[cluster])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 构建并训练GMM模型
k = 3  # 已知有3个簇
gmm = GaussianMixture(n_components=k, covariance_type='full', 
                      max_iter=100, random_state=42)
y_pred = gmm.fit_predict(X_scaled)  # 聚类结果（最可能的簇）
probs = gmm.predict_proba(X_scaled)  # 每个样本属于各个簇的概率

# 输出GMM模型参数
print("高斯混合模型（GMM）结果：")
print(f"迭代次数: {gmm.n_iter_}")
print(f"对数似然值: {gmm.lower_bound_:.4f}")

print("\n每个高斯分量的权重:")
for i in range(k):
    print(f"分量 {i}: {gmm.weights_[i]:.4f}")

print("\n每个高斯分量的均值:")
for i in range(k):
    print(f"分量 {i}: {gmm.means_[i].round(4)}")

# 可视化聚类结果
plt.figure(figsize=(12, 6))

# 1. 聚类结果散点图
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
for i in range(k):
    plt.scatter(X_scaled[y_pred == i, 0], X_scaled[y_pred == i, 1], 
                c=colors[i], alpha=0.6, label=f'簇 {i}')

# 绘制每个高斯分布的均值
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=200, marker='X', 
            c='red', label='均值', edgecolors='black')

plt.title(f'GMM聚类结果 (k={k})')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(alpha=0.3)

# 2. 绘制高斯分布轮廓（椭圆表示）
plt.subplot(1, 2, 2)
for i in range(k):
    # 绘制样本点
    plt.scatter(X_scaled[y_pred == i, 0], X_scaled[y_pred == i, 1], 
                c=colors[i], alpha=0.3)
    
    # 绘制高斯分布的椭圆轮廓（2倍标准差）
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    
    # 计算椭圆参数
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95%置信区间
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                     edgecolor=colors[i], facecolor='none', linewidth=2)
    plt.gca().add_patch(ellipse)

plt.title('GMM高斯分布轮廓（95%置信区间）')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 展示样本的概率分布（选择3个样本）
sample_indices = [10, 100, 200]
plt.figure(figsize=(10, 4))
bar_width = 0.25
for i, idx in enumerate(sample_indices):
    plt.bar(np.arange(k) + i*bar_width, probs[idx], bar_width, 
            label=f'样本 {idx}')

plt.xticks(np.arange(k) + bar_width, [f'簇 {i}' for i in range(k)])
plt.ylabel('概率')
plt.title('样本属于各簇的概率分布')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
    