#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
箱型图检测异常值案例说明：
本案例使用箱型图方法检测数据中的异常值。箱型图基于数据的四分位数，
通过计算上下限（Q1-1.5*IQR和Q3+1.5*IQR）来识别异常值，其中IQR是四分位距。
示例中生成包含异常值的随机数据，使用箱型图进行可视化并标记异常值。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 生成示例数据：主要分布在[10, 50]之间，添加一些异常值
np.random.seed(42)  # 设置随机种子，确保结果可复现
normal_data = np.random.normal(loc=30, scale=8, size=200)  # 正常数据
outliers = np.array([5, 60, 65, 70, -2, 75])  # 异常值
data = np.concatenate([normal_data, outliers])  # 合并数据

# 计算四分位数和异常值边界
q1 = np.percentile(data, 25)  # 第一四分位数
q3 = np.percentile(data, 75)  # 第三四分位数
iqr = q3 - q1  # 四分位距
lower_bound = q1 - 1.5 * iqr  # 下限
upper_bound = q3 + 1.5 * iqr  # 上限

# 检测异常值
outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
outlier_values = data[outlier_indices]

# 输出结果
print("箱型图异常值检测结果：")
print(f"数据总量: {len(data)} 个")
print(f"第一四分位数(Q1): {q1:.2f}")
print(f"第三四分位数(Q3): {q3:.2f}")
print(f"四分位距(IQR): {iqr:.2f}")
print(f"异常值下限: {lower_bound:.2f}")
print(f"异常值上限: {upper_bound:.2f}")
print(f"检测到异常值数量: {len(outlier_values)} 个")
print(f"异常值: {', '.join([f'{x:.2f}' for x in outlier_values])}")

# 可视化箱型图
plt.figure(figsize=(10, 6))
box = plt.boxplot(data, patch_artist=True, 
                 boxprops=dict(facecolor='lightblue', color='blue'),
                 capprops=dict(color='blue'),
                 whiskerprops=dict(color='blue'),
                 flierprops=dict(marker='o', color='red', markersize=8),
                 medianprops=dict(color='green', linewidth=2))

# 添加文本说明
plt.text(1.1, q1, f'Q1: {q1:.2f}', verticalalignment='center')
plt.text(1.1, q3, f'Q3: {q3:.2f}', verticalalignment='center')
plt.text(1.1, lower_bound, f'下限: {lower_bound:.2f}', verticalalignment='center')
plt.text(1.1, upper_bound, f'上限: {upper_bound:.2f}', verticalalignment='center')

plt.title('箱型图异常值检测')
plt.ylabel('数据值')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
    