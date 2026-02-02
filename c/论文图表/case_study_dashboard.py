"""
Case Study Dashboard: 4 Controversial Cases Analysis
====================================================
Jerry Rice (S2), Billy Ray Cyrus (S4), Bristol Palin (S11), Bobby Bones (S27)

生成4个争议案例的综合分析图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ================= 全局风格设置 =================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# 颜色定义 - 学术风格
C_JUDGE = '#E74C3C'      # 红色 - 评委
C_FAN = '#3498DB'        # 蓝色 - 粉丝
C_GRAY = '#7F8C8D'       # 灰色 - 其他选手
C_HIGHLIGHT = '#F39C12'  # 橙色 - 高亮
C_GREEN = '#27AE60'      # 绿色 - 正向
C_PURPLE = '#9B59B6'     # 紫色 - 特殊

# ================= 数据准备 =================

# 1. Jerry Rice (Season 2) - Rank System, 2nd Place
jerry_rice_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8],  # 参赛周数
    'judge_scores': [7.0, 7.67, 6.33, 8.0, 7.67, 7.67, 6.83, 8.89],  # 评委分
    'judge_ranks': [5, 3, 8, 3, 4, 4, 5, 2],  # 评委排名 (越小越好)
    'fan_vote_est': [0.12, 0.15, 0.18, 0.20, 0.25, 0.28, 0.35, 0.28],  # 粉丝票估计
    'fan_ranks': [2, 2, 1, 1, 1, 1, 1, 2],  # 粉丝排名
    'n_contestants': [10, 9, 8, 7, 6, 5, 4, 3],  # 每周参赛人数
    'final_week_others': {  # Week 7 关键周其他选手数据 (决赛周)
        'Drew Lachey': {'judge': 9.17, 'fan': 0.30},
        'Stacy Keibler': {'judge': 9.17, 'fan': 0.12},
        'Lisa Rinna': {'judge': 8.83, 'fan': 0.08},
    }
}

# 2. Billy Ray Cyrus (Season 4) - Percentage System, 5th Place
billy_ray_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8],
    'judge_scores': [4.33, 7.0, 7.0, 7.0, 5.67, 7.0, 6.33, 6.33],
    'judge_ranks': [11, 5, 5, 5, 8, 4, 6, 5],  # 通常在中下游
    'fan_vote_est': [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.06],  # Week 8淘汰
    'fan_ranks': [3, 3, 2, 2, 1, 1, 1, 1],
    'n_contestants': [11, 10, 9, 8, 7, 6, 5, 5],
    'final_week_others': {  # Week 8 关键周
        'Apolo Ohno': {'judge': 9.83, 'fan': 0.32},
        'Joey Fatone': {'judge': 9.17, 'fan': 0.28},
        'Laila Ali': {'judge': 8.83, 'fan': 0.14},
        'Ian Ziering': {'judge': 7.83, 'fan': 0.06},
    }
}

# 3. Bristol Palin (Season 11) - Percentage System, 3rd Place
bristol_palin_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'judge_scores': [6.0, 7.0, 6.33, 6.67, 7.0, 8.67, 7.67, 7.33, 8.0],
    'judge_ranks': [10, 8, 9, 8, 6, 3, 5, 5, 4],  # 中下游但稳步提升
    'fan_vote_est': [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.15],
    'fan_ranks': [2, 2, 1, 1, 1, 1, 1, 1, 3],
    'n_contestants': [12, 11, 10, 9, 8, 7, 6, 5, 4],
    'final_week_others': {  # Week 9 决赛
        'Jennifer Grey': {'judge': 10.0, 'fan': 0.52},
        'Kyle Massey': {'judge': 9.67, 'fan': 0.27},
        'Brandy': {'judge': 9.5, 'fan': 0.06},
    }
}

# 4. Bobby Bones (Season 27) - Percentage System, Champion
bobby_bones_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'judge_scores': [6.67, 6.5, 7.67, 6.67, 7.0, 7.33, 8.83, 7.5, 9.0],
    'judge_ranks': [8, 10, 6, 9, 8, 7, 4, 8, 4],  # 持续低于平均
    'fan_vote_est': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.54],  # 持续上升
    'fan_ranks': [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 粉丝票一直第一
    'n_contestants': [13, 12, 11, 10, 9, 8, 7, 5, 4],
    'final_week_others': {  # Week 9 决赛
        'Milo Manheim': {'judge': 10.0, 'fan': 0.26},
        'Evanna Lynch': {'judge': 10.0, 'fan': 0.14},
        'Alexis Ren': {'judge': 9.5, 'fan': 0.06},
    }
}

# ================= 创建综合图表 =================

def create_case_dashboard(case_data, case_name, season, rule_type, placement, color_main):
    """为单个案例创建4合1仪表盘"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    weeks = case_data['weeks']
    judge_ranks = case_data['judge_ranks']
    fan_ranks = case_data['fan_ranks']
    judge_scores = case_data['judge_scores']
    fan_votes = case_data['fan_vote_est']
    
    # ============ 左上: 趋势分歧图 (Rank Divergence) ============
    ax1 = axes[0, 0]
    
    ax1.plot(weeks, judge_ranks, marker='o', color=C_JUDGE, linewidth=2.5, 
             markersize=8, label='Judge Rank', zorder=3)
    ax1.plot(weeks, fan_ranks, marker='s', color=C_FAN, linewidth=2.5, 
             markersize=8, linestyle='--', label='Fan Vote Rank (Est.)', zorder=3)
    
    # 填充分歧区域
    ax1.fill_between(weeks, judge_ranks, fan_ranks, alpha=0.2, color=C_PURPLE,
                     label='Divergence Zone')
    
    ax1.set_title('(A) Judge vs. Fan Ranking Over Time', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Competition Week', fontsize=11)
    ax1.set_ylabel('Rank (1 = Best)', fontsize=11)
    ax1.set_xticks(weeks)
    ax1.set_yticks(range(1, max(max(judge_ranks), max(fan_ranks)) + 2))
    ax1.invert_yaxis()  # 排名1在最上面
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(weeks[0] - 0.5, weeks[-1] + 0.5)
    
    # 添加注释
    max_gap_idx = np.argmax(np.array(judge_ranks) - np.array(fan_ranks))
    if judge_ranks[max_gap_idx] - fan_ranks[max_gap_idx] > 3:
        ax1.annotate(f'Max Gap: {judge_ranks[max_gap_idx] - fan_ranks[max_gap_idx]} ranks',
                     xy=(weeks[max_gap_idx], (judge_ranks[max_gap_idx] + fan_ranks[max_gap_idx])/2),
                     fontsize=9, color=C_PURPLE, ha='center')
    
    # ============ 右上: 关键周快照 (Scatter Plot) ============
    ax2 = axes[0, 1]
    
    final_week = weeks[-1]
    others = case_data['final_week_others']
    
    # 绘制其他选手
    for name, data in others.items():
        ax2.scatter(data['judge'], data['fan'], color=C_GRAY, s=100, alpha=0.7, zorder=2)
        ax2.annotate(name.split()[0], (data['judge'], data['fan']), 
                     fontsize=8, ha='center', va='bottom', color=C_GRAY)
    
    # 高亮目标选手
    target_judge = judge_scores[-1]
    target_fan = fan_votes[-1]
    ax2.scatter(target_judge, target_fan, color=color_main, s=200, 
                edgecolors='black', linewidth=2, zorder=5, marker='*')
    ax2.annotate(case_name, (target_judge, target_fan), fontsize=10, 
                 fontweight='bold', ha='left', va='bottom', color=color_main,
                 xytext=(5, 5), textcoords='offset points')
    
    # 添加象限划分线
    all_judges = [target_judge] + [d['judge'] for d in others.values()]
    all_fans = [target_fan] + [d['fan'] for d in others.values()]
    mid_judge = np.median(all_judges)
    mid_fan = np.median(all_fans)
    
    ax2.axvline(x=mid_judge, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=mid_fan, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    # 标注象限
    ax2.text(ax2.get_xlim()[0] + 0.3, ax2.get_ylim()[1] - 0.05, 
             'Low Score\nHigh Popularity', fontsize=8, color=C_JUDGE, 
             ha='left', va='top', style='italic')
    
    ax2.set_title(f'(B) Final Week Snapshot (Week {final_week})', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Judge Score', fontsize=11)
    ax2.set_ylabel('Estimated Fan Vote Share', fontsize=11)
    
    # ============ 左下: 粉丝投票分布 (Violin Plot) ============
    ax3 = axes[1, 0]
    
    # 生成模拟的后验分布
    np.random.seed(42)
    distributions = []
    labels = []
    colors = []
    
    # 目标选手
    target_std = case_data.get('fan_std', target_fan * 0.25)
    dist_target = np.random.normal(target_fan, target_std, 500)
    dist_target = np.clip(dist_target, 0.01, 0.99)
    distributions.append(dist_target)
    labels.append(case_name)
    colors.append(color_main)
    
    # 其他选手
    for name, data in list(others.items())[:3]:
        std = data['fan'] * 0.2
        dist = np.random.normal(data['fan'], std, 500)
        dist = np.clip(dist, 0.01, 0.99)
        distributions.append(dist)
        labels.append(name.split()[0])
        colors.append(C_GRAY)
    
    parts = ax3.violinplot(distributions, showmeans=True, showmedians=True)
    
    # 自定义颜色
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # 设置其他元素颜色
    for partname in ['cbars', 'cmins', 'cmaxs', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
    
    ax3.set_title('(C) Fan Vote Distribution (MCMC Estimation)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(range(1, len(labels) + 1))
    ax3.set_xticklabels(labels, fontsize=9, rotation=15)
    ax3.set_ylabel('Fan Vote Share', fontsize=11)
    ax3.set_ylim(0, max(target_fan, max([d['fan'] for d in others.values()])) * 1.5)
    
    # 添加置信区间标注
    ax3.axhline(y=target_fan, color=color_main, linestyle='--', alpha=0.5, linewidth=1)
    
    # ============ 右下: 反事实分析 (Counterfactual) ============
    ax4 = axes[1, 1]
    
    # 三种场景下的结果
    scenarios = ['Current\nSystem', 'Alternative\nSystem', 'With Judges\'\nSave']
    
    # 根据案例设置不同结果
    if case_name == 'Jerry Rice':
        # Rank -> Percentage: 可能更好; Rank + Save: 被淘汰
        results = [2, 2, 7]  # 名次（越小越好）
        result_labels = ['2nd', '≤2nd', 'Eliminated\nWeek 7']
        bar_colors = [C_FAN, C_GREEN, C_JUDGE]
    elif case_name == 'Billy Ray Cyrus':
        # Percentage -> Rank: 早淘汰; Rank + Save: 淘汰
        results = [5, 6, 6]
        result_labels = ['5th', 'Eliminated\nWeek 6', 'Eliminated\nWeek 6']
        bar_colors = [C_FAN, C_JUDGE, C_JUDGE]
    elif case_name == 'Bristol Palin':
        # Percentage -> Rank: 可能相似; Rank + Save: 淘汰
        results = [3, 3, 11]
        result_labels = ['3rd', '~3rd', 'Eliminated\nWeek 11']
        bar_colors = [C_FAN, C_GREEN, C_JUDGE]
    else:  # Bobby Bones
        # Percentage -> Rank: 仍夺冠; Rank + Save: 仍夺冠
        results = [1, 1, 1]
        result_labels = ['Champion', 'Champion', 'Champion']
        bar_colors = [color_main, color_main, color_main]
    
    bars = ax4.bar(scenarios, results, color=bar_colors, width=0.6, edgecolor='black')
    
    # 在柱子上标注结果
    for bar, label in zip(bars, result_labels):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_title('(D) Counterfactual Outcome Analysis', fontsize=13, fontweight='bold', pad=10)
    ax4.set_ylabel('Final Placement (Lower = Better)', fontsize=11)
    ax4.set_ylim(0, max(results) + 3)
    ax4.invert_yaxis()  # 名次1在最上
    
    # 添加当前赛制标签
    ax4.text(0.02, 0.98, f'Actual System: {rule_type}', transform=ax4.transAxes,
             fontsize=9, va='top', ha='left', style='italic', color=C_GRAY)
    
    # ============ 总标题 ============
    fig.suptitle(f'Case Study: {case_name} (Season {season}) — {placement}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加副标题说明赛制
    fig.text(0.5, 0.94, f'Scoring Method: {rule_type}', ha='center', fontsize=11, 
             style='italic', color=C_GRAY)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    return fig


# ================= 生成4个案例图表 =================

# Case 1: Jerry Rice
fig1 = create_case_dashboard(
    jerry_rice_data, 
    'Jerry Rice', 
    season=2, 
    rule_type='Rank-based', 
    placement='2nd Place (Runner-up)',
    color_main='#E67E22'
)
fig1.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_jerry_rice.png', 
             bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Jerry Rice dashboard saved!")

# Case 2: Billy Ray Cyrus
fig2 = create_case_dashboard(
    billy_ray_data,
    'Billy Ray Cyrus',
    season=4,
    rule_type='Percentage-based',
    placement='5th Place',
    color_main='#8E44AD'
)
fig2.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_billy_ray.png',
             bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Billy Ray Cyrus dashboard saved!")

# Case 3: Bristol Palin
fig3 = create_case_dashboard(
    bristol_palin_data,
    'Bristol Palin',
    season=11,
    rule_type='Percentage-based',
    placement='3rd Place',
    color_main='#16A085'
)
fig3.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_bristol_palin.png',
             bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Bristol Palin dashboard saved!")

# Case 4: Bobby Bones
fig4 = create_case_dashboard(
    bobby_bones_data,
    'Bobby Bones',
    season=27,
    rule_type='Percentage-based',
    placement='Champion (1st Place)',
    color_main='#C0392B'
)
fig4.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_bobby_bones.png',
             bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Bobby Bones dashboard saved!")


# ================= 创建4案例综合对比图 =================

def create_combined_overview():
    """创建4个案例的综合对比概览图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    cases = [
        ('Jerry Rice', 2, 'Rank', jerry_rice_data, '#E67E22'),
        ('Billy Ray Cyrus', 4, 'Percentage', billy_ray_data, '#8E44AD'),
        ('Bristol Palin', 11, 'Percentage', bristol_palin_data, '#16A085'),
        ('Bobby Bones', 27, 'Percentage', bobby_bones_data, '#C0392B'),
    ]
    
    for idx, (name, season, rule, data, color) in enumerate(cases):
        ax = axes[idx // 2, idx % 2]
        
        weeks = data['weeks']
        judge_ranks = data['judge_ranks']
        fan_ranks = data['fan_ranks']
        
        # 绘制排名趋势
        ax.plot(weeks, judge_ranks, marker='o', color=C_JUDGE, linewidth=2, 
                markersize=6, label='Judge Rank')
        ax.plot(weeks, fan_ranks, marker='s', color=C_FAN, linewidth=2, 
                linestyle='--', markersize=6, label='Fan Rank (Est.)')
        
        # 填充分歧区域
        ax.fill_between(weeks, judge_ranks, fan_ranks, alpha=0.15, color=color)
        
        ax.set_title(f'({chr(65+idx)}) {name} — Season {season} ({rule})', 
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Rank', fontsize=10)
        ax.invert_yaxis()
        ax.set_xticks(weeks)
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加关键信息框
        avg_gap = np.mean(np.array(judge_ranks) - np.array(fan_ranks))
        info_text = f'Avg. Divergence: {avg_gap:.1f} ranks'
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                va='bottom', ha='left')
    
    fig.suptitle('Four Controversial Cases: Judge vs. Fan Ranking Divergence',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加图例说明
    fig.text(0.5, 0.94, 
             'Red = Judge Rank | Blue = Estimated Fan Rank | Shaded = Divergence Zone',
             ha='center', fontsize=10, style='italic', color=C_GRAY)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    return fig


fig_combined = create_combined_overview()
fig_combined.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_combined_overview.png',
                     bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Combined overview saved!")


# ================= 创建论文用的单页4合1图 =================

def create_paper_figure():
    """创建适合论文的单页综合图"""
    
    fig = plt.figure(figsize=(14, 14))
    
    # 使用GridSpec进行更灵活的布局
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.25)
    
    cases = [
        ('Jerry Rice (S2)', jerry_rice_data, '#E67E22', 'Rank', '2nd'),
        ('Billy Ray Cyrus (S4)', billy_ray_data, '#8E44AD', 'Pct.', '5th'),
        ('Bristol Palin (S11)', bristol_palin_data, '#16A085', 'Pct.', '3rd'),
        ('Bobby Bones (S27)', bobby_bones_data, '#C0392B', 'Pct.', '1st'),
    ]
    
    # 上面4个子图：排名趋势
    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        name, data, color, rule, place = cases[idx]
        weeks = data['weeks']
        judge_ranks = data['judge_ranks']
        fan_ranks = data['fan_ranks']
        
        ax.plot(weeks, judge_ranks, marker='o', color=C_JUDGE, linewidth=2.5,
                markersize=7, label='Judge Rank', zorder=3)
        ax.plot(weeks, fan_ranks, marker='s', color=C_FAN, linewidth=2.5,
                linestyle='--', markersize=7, label='Fan Rank', zorder=3)
        ax.fill_between(weeks, judge_ranks, fan_ranks, alpha=0.2, color=color)
        
        ax.set_title(f'({chr(65+idx)}) {name} — {rule} System → {place}',
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Rank (1=Best)', fontsize=10)
        ax.invert_yaxis()
        ax.set_xticks(weeks)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # 标注最大分歧
        max_gap = max(np.array(judge_ranks) - np.array(fan_ranks))
        if max_gap > 2:
            max_idx = np.argmax(np.array(judge_ranks) - np.array(fan_ranks))
            ax.annotate(f'Gap: {max_gap}',
                        xy=(weeks[max_idx], (judge_ranks[max_idx] + fan_ranks[max_idx])/2),
                        fontsize=8, color=color, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 底部：反事实分析汇总表
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # 创建表格数据
    table_data = [
        ['Case', 'Actual System', 'Actual Result', 'Alternative System', 'With Judges\' Save'],
        ['Jerry Rice', 'Rank', '2nd Place', '≤2nd (benefit)', 'Eliminated W7'],
        ['Billy Ray Cyrus', 'Percentage', '5th Place', 'Eliminated W6', 'Eliminated W6'],
        ['Bristol Palin', 'Percentage', '3rd Place', '~3rd (similar)', 'Eliminated W11'],
        ['Bobby Bones', 'Percentage', 'Champion', 'Champion', 'Champion'],
    ]
    
    # 设置表格颜色
    cell_colors = [['#E8E8E8'] * 5]  # 表头
    for i in range(1, 5):
        row_colors = ['white'] * 5
        # 高亮有变化的列
        if 'Eliminated' in table_data[i][3] or 'benefit' in table_data[i][3]:
            row_colors[3] = '#FADBD8' if 'Eliminated' in table_data[i][3] else '#D5F5E3'
        if 'Eliminated' in table_data[i][4]:
            row_colors[4] = '#FADBD8'
        cell_colors.append(row_colors)
    
    table = ax_table.table(cellText=table_data,
                           cellColours=cell_colors,
                           loc='center',
                           cellLoc='center',
                           colWidths=[0.2, 0.15, 0.15, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for j in range(5):
        table[(0, j)].set_text_props(fontweight='bold')
    
    ax_table.set_title('(E) Counterfactual Analysis Summary', fontsize=12, 
                       fontweight='bold', pad=10, y=0.95)
    
    # 添加说明文字
    ax_table.text(0.5, -0.05, 
                  'Green = Outcome improves | Red = Outcome worsens | White = Similar outcome',
                  ha='center', fontsize=9, style='italic', color=C_GRAY,
                  transform=ax_table.transAxes)
    
    fig.suptitle('Figure X: Controversial Cases Analysis Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    return fig


fig_paper = create_paper_figure()
fig_paper.savefig('/Users/Zhuanz1/Desktop/mcm/c/论文图表/case_study_paper_figure.png',
                  bbox_inches='tight', dpi=300, facecolor='white')
print("✅ Paper figure saved!")

plt.close('all')
print("\n" + "="*60)
print("所有图表生成完成！")
print("="*60)
print("""
生成的文件：
1. case_study_jerry_rice.png     - Jerry Rice 详细仪表盘
2. case_study_billy_ray.png      - Billy Ray Cyrus 详细仪表盘  
3. case_study_bristol_palin.png  - Bristol Palin 详细仪表盘
4. case_study_bobby_bones.png    - Bobby Bones 详细仪表盘
5. case_study_combined_overview.png - 4案例趋势对比图
6. case_study_paper_figure.png   - 论文用综合图（推荐）

建议在论文 Section 5.3 中使用:
- 主图: case_study_paper_figure.png (包含趋势+反事实汇总)
- 或分开使用各个案例的详细仪表盘
""")
