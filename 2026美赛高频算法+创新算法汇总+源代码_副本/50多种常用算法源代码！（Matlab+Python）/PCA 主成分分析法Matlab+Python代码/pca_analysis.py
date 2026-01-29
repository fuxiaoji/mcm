#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA主成分分析法案例说明：
本案例使用主成分分析(PCA)对高维数据进行降维。PCA是一种常用的降维技术，
通过线性变换将高维数据映射到低维空间，同时保留数据中最重要的信息。
示例中使用鸢尾花数据集(4维特征)，通过PCA降维到2维空间进行可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 加载数据集
iris = load_iris()
X = iris.data  # 4维特征数据
y = iris.target  # 标签
feature_names = iris.feature_names
target_names = iris.target_names

# 数据标准化（PCA对数据尺度敏感，通常需要标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 输出PCA结果分析
print("PCA主成分分析结果：")
print(f"原始数据维度: {X.shape[1]} 维")
print(f"降维后数据维度: {X_pca.shape[1]} 维")
print("\n主成分解释方差比例:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
print(f"\n累计解释方差比例: {sum(pca.explained_variance_ratio_):.4f} "
      f"({sum(pca.explained_variance_ratio_)*100:.2f}%)")

print("\n主成分载荷矩阵（表示原始特征与主成分的相关性）:")
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(
    loadings, 
    columns=[f'主成分 {i+1}' for i in range(2)],
    index=feature_names
)
print(loading_matrix.round(4))

# 可视化PCA降维结果
plt.figure(figsize=(10, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA降维：鸢尾花数据集(4维→2维)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 绘制解释方差比例条形图
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.5, 
        align='center',
        label='单个主成分解释方差')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 
         where='mid',
         label='累计解释方差')
plt.ylabel('解释方差比例')
plt.xlabel('主成分数量')
plt.title('PCA解释方差比例')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 为了使代码可运行，需要导入pandas
import pandas as pd
    