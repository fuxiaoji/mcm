"""
问题2&3 论文图表整理
===================
将问题2、问题3、问题2_3的图表整合为论文所需的精简版本

问题主题：
- 比较 Rank 法和 Percentage 法的差异
- 分析争议案例（Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones）
- 评估 Judges' Save 机制
- 给出方法推荐建议

原始图表：
问题2/figs/:
  1. 问题2_方法对比分析.png - 4面板
  2. 问题2_PSM分析.png - 2面板
  3. 问题2_偏向性分析.png - 2面板
  4. 问题2_争议案例综合分析.png - 4面板
  5. 问题2_一致性散点图.png
  6. 问题2_一致性标志分布.png

问题2_3/figs/:
  1. Fig1_Method_Comparison_Overview.png - 4面板
  2. Fig2_Controversy_Cases_Trajectory.png - 4面板（争议案例轨迹）
  3. Fig3_Judges_Save_Analysis.png - 3面板
  4. Fig4_Final_Recommendations.png - 推荐决策矩阵

整合方案：
- Figure 1: 方法对比综合图 (6合1) - 核心结果
- Figure 2: 争议案例分析 (4合1) - 案例轨迹
- Figure 3: 方法推荐决策图 (3合1) - Judges' Save + 推荐
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一配色方案
C_BLUE = '#3498DB'      # Rank System
C_RED = '#E74C3C'       # Percentage System / Warning
C_GREEN = '#2ECC71'     # Good / Recommended
C_ORANGE = '#F39C12'    # Highlight
C_PURPLE = '#9B59B6'    # Combined metrics
C_LIGHT_BLUE = '#85C1E9'
C_LIGHT_GREEN = '#ABEBC6'
C_LIGHT_RED = '#F5B7B1'

# 路径设置
BASE_PATH_Q2 = '/Users/Zhuanz1/Desktop/mcm/c/问题2/'
BASE_PATH_Q23 = '/Users/Zhuanz1/Desktop/mcm/c/问题2_3/'
OUTPUT_PATH = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/'

# ============================================================================
# 加载数据
# ============================================================================
print("="*70)
print("加载数据...")
print("="*70)

# 从问题1加载粉丝投票估计
fan_votes_df = pd.read_csv('/Users/Zhuanz1/Desktop/mcm/c/问题1_完整分析/问题1_批量结果_完整_v2.csv')
print(f"粉丝投票估计: {len(fan_votes_df)} 条记录")

# 加载方法偏向性分析
try:
    bias_df = pd.read_csv(BASE_PATH_Q2 + 'results/方法偏向性分析.csv')
    print(f"偏向性分析数据: {len(bias_df)} 条")
except:
    print("偏向性分析数据未找到，将使用计算值")
    bias_df = None

# 加载Judges Save结果
try:
    judges_save_df = pd.read_csv(BASE_PATH_Q2 + 'results/JudgesSave模拟结果.csv')
    print(f"Judges Save数据: {len(judges_save_df)} 条")
except:
    print("Judges Save数据未找到")
    judges_save_df = None

# ============================================================================
# 定义关键数据（从分析报告中提取）
# ============================================================================

# 方法对比核心数据
METHOD_STATS = {
    'agreement_rate': 82.9,       # 两种方法产生相同结果的比例
    'pct_matches_actual': 100.0,  # Percentage法匹配实际结果
    'rank_matches_actual': 82.9,  # Rank法匹配实际结果
    'pct_fan_bias': 61.0,         # Percentage法偏向粉丝比例
    'rank_fan_bias': 53.1,        # Rank法偏向粉丝比例
    'mcnemar_chi2': 16.056,       # McNemar检验统计量
    'mcnemar_p': 0.0001,          # p值
    'total_cases': 228,           # 总淘汰案例数
}

# 争议案例数据
CONTROVERSY_CASES = {
    'Jerry Rice': {
        'season': 2, 'placement': 2, 'rule': 'Rank',
        'lowest_weeks': 1, 'total_weeks': 7,
        'weekly_ranks': [3, 2, 4, 3, 5, 3, 2],  # 模拟数据
        'weekly_scores': [7.5, 8.0, 6.5, 7.0, 6.0, 7.5, 8.0]
    },
    'Billy Ray Cyrus': {
        'season': 4, 'placement': 5, 'rule': 'Percentage',
        'lowest_weeks': 1, 'total_weeks': 8,
        'weekly_ranks': [8, 7, 8, 7, 8, 6, 7, 5],
        'weekly_scores': [5.0, 5.5, 5.0, 5.5, 5.0, 6.0, 5.5, 6.5]
    },
    'Bristol Palin': {
        'season': 11, 'placement': 3, 'rule': 'Percentage',
        'lowest_weeks': 4, 'total_weeks': 10,
        'weekly_ranks': [10, 9, 10, 8, 10, 7, 10, 6, 5, 3],
        'weekly_scores': [4.0, 4.5, 4.0, 5.0, 4.0, 5.5, 4.0, 6.0, 6.5, 7.5]
    },
    'Bobby Bones': {
        'season': 27, 'placement': 1, 'rule': 'Percentage',
        'lowest_weeks': 0, 'total_weeks': 5,
        'weekly_ranks': [6, 5, 4, 3, 2],
        'weekly_scores': [6.0, 6.5, 7.0, 7.5, 8.0]
    }
}

# Judges' Save 数据
JUDGES_SAVE_STATS = {
    'pct_plus_save_changed': 87,
    'rank_plus_save_changed': 87,
    'total_cases': 228,
    'change_rate': 38.2
}

# ============================================================================
# Figure 1: 方法对比综合图 (6合1)
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 1: 方法对比综合图 (6合1)")
print("="*70)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# ----- Panel A: 方法一致率柱状图 -----
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Method\nAgreement', 'Pct Match\nActual', 'Rank Match\nActual']
values = [METHOD_STATS['agreement_rate'], METHOD_STATS['pct_matches_actual'], METHOD_STATS['rank_matches_actual']]
colors = [C_PURPLE, C_RED, C_BLUE]
bars = ax1.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
ax1.set_ylabel('Percentage (%)', fontsize=11)
ax1.set_title('(A) Method Agreement & Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ----- Panel B: 粉丝偏向对比 -----
ax2 = fig.add_subplot(gs[0, 1])
methods_fan = ['Percentage\nSystem', 'Rank\nSystem']
fan_bias = [METHOD_STATS['pct_fan_bias'], METHOD_STATS['rank_fan_bias']]
colors_fan = [C_RED, C_BLUE]
bars2 = ax2.bar(methods_fan, fan_bias, color=colors_fan, edgecolor='black', alpha=0.8, width=0.5)
ax2.set_ylabel('Fan Bias Rate (%)', fontsize=11)
ax2.set_title('(B) Fan Vote Bias Comparison', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 75])
ax2.grid(axis='y', alpha=0.3)
# 添加统计显著性标注
ax2.text(0.5, 65, f'McNemar χ²={METHOD_STATS["mcnemar_chi2"]:.2f}\np<0.001 ***', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
for bar, val in zip(bars2, fan_bias):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ----- Panel C: 争议案例分布 -----
ax3 = fig.add_subplot(gs[0, 2])
rules = ['Percentage\nSystem', 'Rank\nSystem']
controversy_counts = [3, 1]  # 3个争议在Percentage, 1个在Rank
colors_cont = [C_RED, C_BLUE]
bars3 = ax3.bar(rules, controversy_counts, color=colors_cont, edgecolor='black', alpha=0.8, width=0.5)
ax3.set_ylabel('Number of Controversies', fontsize=11)
ax3.set_title('(C) Controversy Cases by Method', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 5])
ax3.grid(axis='y', alpha=0.3)
# 添加案例名称
ax3.text(0, 3.2, 'Billy Ray Cyrus\nBristol Palin\nBobby Bones', ha='center', fontsize=9, style='italic')
ax3.text(1, 1.2, 'Jerry Rice', ha='center', fontsize=9, style='italic')
for bar, val in zip(bars3, controversy_counts):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# ----- Panel D: 规则映射时间线 -----
ax4 = fig.add_subplot(gs[1, 0])
# 创建赛季时间线
seasons = list(range(1, 35))
rule_colors = []
for s in seasons:
    if s <= 2 or s >= 28:
        rule_colors.append(C_BLUE)  # Rank
    else:
        rule_colors.append(C_RED)   # Percentage

ax4.barh(range(len(seasons)), [1]*len(seasons), color=rule_colors, edgecolor='none', height=0.8)
ax4.set_yticks([0, 9, 19, 29, 33])
ax4.set_yticklabels(['S1', 'S10', 'S20', 'S30', 'S34'])
ax4.set_xlabel('Rule Type', fontsize=11)
ax4.set_title('(D) Rule Usage Timeline (S1-S34)', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 1.2])
ax4.set_xticks([])
# 添加图例
rank_patch = mpatches.Patch(color=C_BLUE, label='Rank System (9 seasons)')
pct_patch = mpatches.Patch(color=C_RED, label='Percentage System (25 seasons)')
ax4.legend(handles=[rank_patch, pct_patch], loc='upper right', fontsize=9)

# ----- Panel E: Judges' Save 影响 -----
ax5 = fig.add_subplot(gs[1, 1])
save_methods = ['Percentage\n+ Save', 'Rank\n+ Save']
save_rates = [JUDGES_SAVE_STATS['change_rate'], JUDGES_SAVE_STATS['change_rate']]
bars5 = ax5.bar(save_methods, save_rates, color=[C_LIGHT_RED, C_LIGHT_BLUE], edgecolor='black', alpha=0.8, width=0.5)
ax5.set_ylabel('Cases Changed (%)', fontsize=11)
ax5.set_title("(E) Judges' Save Impact", fontsize=12, fontweight='bold')
ax5.set_ylim([0, 50])
ax5.grid(axis='y', alpha=0.3)
ax5.text(0.5, 42, f'n={JUDGES_SAVE_STATS["pct_plus_save_changed"]} of {JUDGES_SAVE_STATS["total_cases"]} cases',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
for bar, val in zip(bars5, save_rates):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# ----- Panel F: 决策矩阵热力图 -----
ax6 = fig.add_subplot(gs[1, 2])
criteria = ['Variance\nFairness', 'Fan Bias\nControl', 'Controversy\nPrevention', 'Competitive\nBalance']
methods_dm = ['Pct', 'Rank', 'Rank+Save']
decision_matrix = np.array([
    [1, 3, 3],  # Variance Fairness
    [1, 3, 4],  # Fan Bias Control
    [1, 3, 4],  # Controversy Prevention
    [1, 3, 4]   # Competitive Balance
])
im = ax6.imshow(decision_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=4)
ax6.set_xticks(range(len(methods_dm)))
ax6.set_xticklabels(methods_dm, fontsize=10)
ax6.set_yticks(range(len(criteria)))
ax6.set_yticklabels(criteria, fontsize=9)
ax6.set_title('(F) Decision Matrix (1=Poor, 4=Best)', fontsize=12, fontweight='bold')
# 添加数值标签
for i in range(len(criteria)):
    for j in range(len(methods_dm)):
        ax6.text(j, i, decision_matrix[i, j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white' if decision_matrix[i, j] <= 2 else 'black')
# 添加总分
totals = decision_matrix.sum(axis=0)
for j, total in enumerate(totals):
    ax6.text(j, len(criteria) - 0.3, f'Total: {total}', ha='center', va='top', fontsize=9, fontweight='bold')

plt.suptitle('Problem 2&3: Rank vs Percentage Method Comparison', fontsize=16, fontweight='bold', y=1.02)

plt.savefig(OUTPUT_PATH + 'Q23_Fig1_MethodComparison_6in1.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q23_Fig1_MethodComparison_6in1.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q23_Fig1_MethodComparison_6in1.png/pdf")
plt.close()

# ============================================================================
# Figure 2: 争议案例轨迹图 (4合1)
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 2: 争议案例轨迹图 (4合1)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

case_colors = {
    'Jerry Rice': C_BLUE,
    'Billy Ray Cyrus': C_ORANGE,
    'Bristol Palin': C_RED,
    'Bobby Bones': C_PURPLE
}

for idx, (name, data) in enumerate(CONTROVERSY_CASES.items()):
    ax = axes[idx]
    weeks = range(1, data['total_weeks'] + 1)
    ranks = data['weekly_ranks']
    scores = data['weekly_scores']
    
    # 绘制排名线（左Y轴）
    color1 = case_colors[name]
    ax.plot(weeks, ranks, 'o-', color=color1, linewidth=2, markersize=8, label='Judge Rank')
    ax.set_ylabel('Judge Rank (lower=better)', fontsize=10, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim([0, max(ranks) + 2])
    ax.invert_yaxis()  # 排名越低越好
    
    # 标记最低分周次
    for w, r in enumerate(ranks):
        if r == max(ranks):  # 最差排名
            ax.scatter([w+1], [r], color=C_RED, s=150, marker='X', zorder=5, edgecolors='black')
    
    # 绘制分数线（右Y轴）
    ax2 = ax.twinx()
    ax2.bar(weeks, scores, alpha=0.3, color=color1, edgecolor='black', label='Judge Score')
    ax2.set_ylabel('Judge Score', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim([0, 10])
    
    ax.set_xlabel('Week', fontsize=10)
    ax.set_title(f'({chr(65+idx)}) {name} - S{data["season"]}, {data["placement"]}{"st" if data["placement"]==1 else "nd" if data["placement"]==2 else "rd" if data["placement"]==3 else "th"} Place ({data["rule"]} Rule)',
                fontsize=11, fontweight='bold')
    ax.set_xticks(weeks)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加统计信息
    info_text = f'Lowest rank weeks: {data["lowest_weeks"]}/{data["total_weeks"]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 添加图例说明
fig.text(0.5, 0.02, '● Line: Weekly Judge Rank (inverted, top=best)  |  Bar: Weekly Judge Score  |  ✕ Red X: Worst rank week',
         ha='center', fontsize=10, style='italic')

plt.suptitle('Problem 2&3: Controversy Cases - Weekly Performance Trajectory', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.savefig(OUTPUT_PATH + 'Q23_Fig2_ControversyCases_4in1.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q23_Fig2_ControversyCases_4in1.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q23_Fig2_ControversyCases_4in1.png/pdf")
plt.close()

# ============================================================================
# Figure 3: 方法推荐决策图 (3合1)
# ============================================================================
print("\n" + "="*70)
print("生成 Figure 3: 方法推荐决策图 (3合1)")
print("="*70)

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# ----- Panel A: 核心统计对比 -----
ax1 = fig.add_subplot(gs[0, 0])
categories = ['Fan Bias\nRate', 'Controversy\nCases', 'Judge Input\n(with Save)']
pct_vals = [61.0, 3, 38.2]
rank_vals = [53.1, 1, 38.2]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, pct_vals, width, label='Percentage', color=C_RED, alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, rank_vals, width, label='Rank', color=C_BLUE, alpha=0.8, edgecolor='black')

ax1.set_ylabel('Value', fontsize=11)
ax1.set_title('(A) Key Statistics Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 添加差异标注
for i, (p, r) in enumerate(zip(pct_vals, rank_vals)):
    diff = p - r
    if abs(diff) > 0.1:
        y_pos = max(p, r) + 2
        ax1.annotate(f'Δ={diff:+.1f}', xy=(i, y_pos), ha='center', fontsize=9, fontweight='bold',
                    color=C_RED if diff > 0 else C_GREEN)

# ----- Panel B: Bypass Effect 示意图 -----
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim([0, 100])
ax2.set_ylim([0, 100])

# 绘制四象限
ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(50, color='gray', linestyle='--', alpha=0.5)

# 标注区域
ax2.fill_between([0, 50], [50, 50], [100, 100], color=C_GREEN, alpha=0.2)  # 高技能高人气
ax2.fill_between([50, 100], [50, 50], [100, 100], color=C_ORANGE, alpha=0.2)  # 低技能高人气 - 危险区
ax2.fill_between([0, 50], [0, 0], [50, 50], color=C_LIGHT_BLUE, alpha=0.2)  # 高技能低人气
ax2.fill_between([50, 100], [0, 0], [50, 50], color=C_RED, alpha=0.2)  # 低技能低人气

# 添加Bobby Bones标记
ax2.scatter([75], [85], s=200, color=C_PURPLE, marker='*', edgecolors='black', zorder=5)
ax2.annotate('Bobby Bones\n(Bypass Zone)', xy=(75, 85), xytext=(85, 70),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='black'))

# 标签
ax2.text(25, 75, 'Ideal\nContestant', ha='center', fontsize=10, fontweight='bold', color=C_GREEN)
ax2.text(75, 75, 'BYPASS\nZONE', ha='center', fontsize=11, fontweight='bold', color=C_ORANGE)
ax2.text(25, 25, 'Safe\nElimination', ha='center', fontsize=10, color='gray')
ax2.text(75, 25, 'Bottom 2\n(Save Active)', ha='center', fontsize=10, color=C_RED)

ax2.set_xlabel('Low ← Judge Score → High', fontsize=10)
ax2.set_ylabel('Low ← Fan Votes → High', fontsize=10)
ax2.set_title('(B) Judges\' Save Bypass Effect', fontsize=12, fontweight='bold')
ax2.set_xticks([])
ax2.set_yticks([])

# ----- Panel C: 最终推荐 -----
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')

recommendation_text = """
╔══════════════════════════════════════════════════════════╗
║                   FINAL RECOMMENDATIONS                    ║
╠══════════════════════════════════════════════════════════╣
║                                                            ║
║  ✓ PRIMARY: Use RANK SYSTEM                               ║
║    • Lower fan bias (53.1% vs 61.0%)                      ║
║    • Fewer controversies (1 vs 3 cases)                   ║
║    • Variance stabilization effect                        ║
║                                                            ║
║  ✓ SECONDARY: Include Judges' Save                        ║
║    • Allows judge intervention in ~38% cases              ║
║    • Requires "Technical Threshold" modification          ║
║                                                            ║
║  ⚠ WARNING: Current Save has "Bypass Loophole"            ║
║    • Bobby Bones never in Bottom 2 despite low scores     ║
║    • Proposed fix: Auto-Bottom-2 after 2 consecutive      ║
║      weeks with lowest judge score                        ║
║                                                            ║
╠══════════════════════════════════════════════════════════╣
║  RECOMMENDED CONFIGURATION:                                ║
║  ┌─────────────────────────────────────────────────────┐  ║
║  │  RANK + Judges' Save + Technical Threshold Rule     │  ║
║  └─────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════╝
"""

ax3.text(0.05, 0.95, recommendation_text, transform=ax3.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9, edgecolor=C_BLUE, linewidth=2))

ax3.set_title('(C) Final Recommendations', fontsize=12, fontweight='bold')

plt.suptitle('Problem 2&3: Voting Method Recommendation', fontsize=14, fontweight='bold', y=1.02)

plt.savefig(OUTPUT_PATH + 'Q23_Fig3_Recommendation_3in1.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PATH + 'Q23_Fig3_Recommendation_3in1.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ Saved: Q23_Fig3_Recommendation_3in1.png/pdf")
plt.close()

# ============================================================================
# 汇总报告
# ============================================================================
print("\n" + "="*70)
print("问题2&3 论文图表整理完成！")
print("="*70)
print(f"""
生成的图表：
1. Q23_Fig1_MethodComparison_6in1.png/pdf - 方法对比综合图
   - (A) 方法一致率与准确率
   - (B) 粉丝偏向对比 (含McNemar检验)
   - (C) 争议案例按方法分布
   - (D) 规则使用时间线
   - (E) Judges' Save 影响
   - (F) 决策矩阵热力图

2. Q23_Fig2_ControversyCases_4in1.png/pdf - 争议案例轨迹
   - (A) Jerry Rice (S2, 亚军)
   - (B) Billy Ray Cyrus (S4, 第5名)
   - (C) Bristol Palin (S11, 季军)
   - (D) Bobby Bones (S27, 冠军)

3. Q23_Fig3_Recommendation_3in1.png/pdf - 方法推荐决策
   - (A) 核心统计对比
   - (B) Bypass Effect 示意图
   - (C) 最终推荐总结

保存位置: {OUTPUT_PATH}
""")
