"""
问题1 论文图表整理
===================
将问题1的7张图表整合为论文所需的精简版本

原始图表清单：
1. 问题1_综合分析图表.png - 4面板：确定性分布/赛季/一致性/散点
2. 问题1_综合分析图表_v2.png - 更新的4面板版本
3. 问题1_评分离散度分析.png - 4面板：离散度分析
4. 问题1_一致性定义对比.png - 一致性方法对比
5. 问题1_示例分析.png - 案例展示
6. 问题1_数据驱动分析.png - 数据驱动分析
7. 问题1_模型评估.png - 模型评估

整合方案：
- Figure 1: 模型框架 + 方法论（需要新建）
- Figure 2: 核心结果 6合1（确定性分布 + 赛季变化 + 离散度相关性）
- Figure 3: 案例展示（保留原图或2合1）
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一配色方案
C_BLUE = '#3498DB'
C_RED = '#E74C3C'
C_GREEN = '#2ECC71'
C_ORANGE = '#F39C12'
C_PURPLE = '#9B59B6'
C_LIGHT_BLUE = '#85C1E9'

# 路径设置
BASE_PATH = '/Users/Zhuanz1/Desktop/mcm/c/问题1_完整分析/'
OUTPUT_PATH = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/'

# ============================================================================
# 读取数据（需要从notebook中提取或重新计算）
# ============================================================================
print("="*70)
print("加载数据...")
print("="*70)

# 加载结果数据
results_df = pd.read_csv(BASE_PATH + '问题1_批量结果_完整_v2.csv')
print(f"加载 {len(results_df)} 条记录")

# ============================================================================
# Figure 1: 问题1核心结果 - 6合1大图
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 1: 问题1核心结果汇总 (6合1)")
print("="*70)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

# ----- Panel A: 确定性指数分布 -----
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(results_df['certainty'], bins=30, color=C_BLUE, edgecolor='black', alpha=0.7)
mean_cert = results_df['certainty'].mean()
median_cert = results_df['certainty'].median()
ax1.axvline(mean_cert, color=C_RED, linestyle='--', linewidth=2, label=f'Mean: {mean_cert:.3f}')
ax1.axvline(median_cert, color=C_GREEN, linestyle='--', linewidth=2, label=f'Median: {median_cert:.3f}')
ax1.set_xlabel('Certainty Index', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('(A) Distribution of Certainty', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# ----- Panel B: 赛季变化箱线图 -----
ax2 = fig.add_subplot(gs[0, 1])
seasons = sorted(results_df['season'].unique())
season_data = [results_df[results_df['season'] == s]['certainty'].values for s in seasons]
bp = ax2.boxplot(season_data, labels=[f'{s}' for s in seasons],
                  patch_artist=True, notch=False, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(C_LIGHT_BLUE)
    patch.set_alpha(0.7)
ax2.set_xlabel('Season', fontsize=11)
ax2.set_ylabel('Certainty Index', fontsize=11)
ax2.set_title('(B) Certainty by Season', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticklabels([f'{s}' if i % 3 == 0 else '' for i, s in enumerate(seasons)], fontsize=8)

# ----- Panel C: 一致性率按参与人数 -----
ax3 = fig.add_subplot(gs[0, 2])
# 创建参与人数分组
results_df['size_group'] = pd.cut(results_df['n_participants'], 
                                   bins=[0, 5, 8, 11, 20], 
                                   labels=['2-5', '6-8', '9-11', '12+'])
size_groups = results_df.groupby('size_group', observed=True)['rank_consistency'].agg(['mean', 'count']).reset_index()
bars = ax3.bar(range(len(size_groups)), size_groups['mean'], color=C_RED, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Participant Group Size', fontsize=11)
ax3.set_ylabel('Consistency Rate', fontsize=11)
ax3.set_title('(C) Consistency by Group Size', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(size_groups)))
ax3.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(size_groups['size_group'], size_groups['count'])], fontsize=9)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, size_groups['mean']):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.1%}', ha='center', va='bottom', fontsize=9)

# ----- Panel D: 评委分数变异系数分布 -----
ax4 = fig.add_subplot(gs[1, 0])
# 计算judge_mean (从其他列推算)
if 'judge_mean' not in results_df.columns:
    results_df['judge_mean'] = results_df['eliminated_judge_score']  # 使用淘汰者分数作为代理
results_df['judge_cv'] = results_df['judge_std'] / results_df['judge_mean']
ax4.hist(results_df['judge_cv'].dropna(), bins=30, color=C_ORANGE, edgecolor='black', alpha=0.7)
mean_cv = results_df['judge_cv'].mean()
ax4.axvline(mean_cv, color=C_RED, linestyle='--', linewidth=2, label=f'Mean: {mean_cv:.4f}')
ax4.set_xlabel('Judge Score CV', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('(D) Judge Score Dispersion', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# ----- Panel E: 离散度 vs 确定性散点图 -----
ax5 = fig.add_subplot(gs[1, 1])
from scipy.stats import pearsonr
scatter = ax5.scatter(results_df['judge_cv'], results_df['certainty'], 
                      c=results_df['rank_consistency'], cmap='RdYlGn', 
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.3)
r, p = pearsonr(results_df['judge_cv'].dropna(), results_df['certainty'].dropna())
ax5.set_xlabel('Judge Score CV', fontsize=11)
ax5.set_ylabel('Certainty Index', fontsize=11)
ax5.set_title('(E) Dispersion vs Certainty', fontsize=12, fontweight='bold')
ax5.text(0.05, 0.95, f'r = {r:.3f}' + ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''),
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.colorbar(scatter, ax=ax5, label='Consistency')
ax5.grid(alpha=0.3)

# ----- Panel F: 参与人数 vs 确定性 -----
ax6 = fig.add_subplot(gs[1, 2])
scatter2 = ax6.scatter(results_df['n_participants'], results_df['certainty'], 
                       c=results_df['rank_consistency'], cmap='RdYlGn', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.3)
ax6.set_xlabel('Number of Participants', fontsize=11)
ax6.set_ylabel('Certainty Index', fontsize=11)
ax6.set_title('(F) Participants vs Certainty', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax6, label='Consistency')
ax6.grid(alpha=0.3)

plt.suptitle('Problem 1: Fan Vote Reconstruction Model - Key Results', 
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig(OUTPUT_PATH + 'Q1_Fig1_KeyResults_6in1.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q1_Fig1_KeyResults_6in1.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q1_Fig1_KeyResults_6in1.png/pdf")
plt.close()

# ============================================================================
# Figure 2: 一致性定义对比 + 方法论图
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 2: 方法论对比图 (2合1)")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: 一致性方法对比条形图
ax = axes[0]
methods = ['Rank-Based\nMethod', 'Convex\nOptimization']
consistency_rates = [37.3, 100.0]
colors = [C_RED, C_GREEN]
bars = ax.bar(methods, consistency_rates, color=colors, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
ax.set_ylabel('Consistency Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('(A) Consistency Method Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([0, 115])
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, consistency_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Panel B: 核心统计汇总
ax = axes[1]
ax.axis('off')

# 创建统计表格
stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║             Fan Vote Reconstruction Model Summary              ║
╠══════════════════════════════════════════════════════════════╣
║  Total Voting Events:           435                           ║
║  - Elimination Events:          324                           ║
║  - Finalist Reconstructions:    111                           ║
║  Contestant Coverage:           395/408 (96.8%)               ║
╠══════════════════════════════════════════════════════════════╣
║  Certainty Index                                               ║
║  - Mean:                        {results_df['certainty'].mean():.4f}                          ║
║  - Median:                      {results_df['certainty'].median():.4f}                          ║
║  - Std:                         {results_df['certainty'].std():.4f}                          ║
╠══════════════════════════════════════════════════════════════╣
║  Rank Consistency Rate:         {results_df['rank_consistency'].mean()*100:.1f}%                          ║
║  Convex Constraint Feasibility: 100.0%                        ║
╚══════════════════════════════════════════════════════════════╝
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Q1_Fig2_Methodology_2in1.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q1_Fig2_Methodology_2in1.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q1_Fig2_Methodology_2in1.png/pdf")
plt.close()

# ============================================================================
# Figure 3: 案例展示 - 典型案例分析
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 3: 典型案例展示")
print("="*70)

# 选取高确定性和低确定性的典型案例
high_cert_case = results_df.nlargest(1, 'certainty').iloc[0]
low_cert_case = results_df.nsmallest(1, 'certainty').iloc[0]
median_cert_case = results_df.iloc[(results_df['certainty'] - results_df['certainty'].median()).abs().argsort()[:1]].iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cases = [
    (high_cert_case, 'High Certainty Case', C_GREEN),
    (median_cert_case, 'Median Certainty Case', C_BLUE),
    (low_cert_case, 'Low Certainty Case', C_RED)
]

for idx, (case, title, color) in enumerate(cases):
    ax = axes[idx]
    # 创建案例信息展示
    judge_range = case.get('judge_range', case['judge_std']*2 if 'judge_std' in case else 0)
    info_text = f"""
Season: {case['season']}, Week: {case['week']}
Eliminated: {case['eliminated_name'] if pd.notna(case.get('eliminated_name', None)) else 'N/A'}

Certainty Index: {case['certainty']:.4f}
Participants: {case['n_participants']}
Judge Score Std: {case.get('judge_std', 0):.2f}

Rank Consistency: {'Yes' if case['rank_consistency'] == 1 else 'No'}
"""
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    ax.set_title(f'({chr(65+idx)}) {title}', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.suptitle('Problem 1: Representative Case Studies', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'Q1_Fig3_CaseStudies.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q1_Fig3_CaseStudies.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q1_Fig3_CaseStudies.png/pdf")
plt.close()

# ============================================================================
# 汇总报告
# ============================================================================
print("\n" + "="*70)
print("问题1 论文图表整理完成！")
print("="*70)
print(f"""
生成的图表：
1. Q1_Fig1_KeyResults_6in1.png/pdf - 核心结果6合1大图
   - (A) 确定性指数分布
   - (B) 赛季变化箱线图
   - (C) 一致性率按组大小
   - (D) 评委分数变异系数分布
   - (E) 离散度 vs 确定性散点图
   - (F) 参与人数 vs 确定性

2. Q1_Fig2_Methodology_2in1.png/pdf - 方法论对比
   - (A) 一致性方法对比
   - (B) 模型统计汇总

3. Q1_Fig3_CaseStudies.png/pdf - 典型案例展示

保存位置: {OUTPUT_PATH}
""")
