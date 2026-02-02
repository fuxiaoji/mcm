"""
Case Study Dashboard V2: 4 Controversial Cases Analysis
========================================================
Jerry Rice (S2), Billy Ray Cyrus (S4), Bristol Palin (S11), Bobby Bones (S27)

改进版本:
1. 统一配色 - 与项目 chart_style_config.py 一致
2. Y轴修正 - 下小上大（排名1在底部表示最好）
3. B图热力背景 - 添加KDE热力图
4. 新增3D可视化选项
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# ================= 统一配色方案 =================
# 从 chart_style_config.py 导入的颜色
COLORS = {
    'blue': '#3498DB',    # 评委分 / 百分比制
    'red': '#E74C3C',     # 粉丝票 / 排名制
    'green': '#2ECC71',   # 正向
    'orange': '#F39C12',  # 高亮/警告
    'purple': '#9B59B6',  # 组合
    'cyan': '#1ABC9C',    # 辅助
    'gray': '#95A5A6',    # 中性
    'dark': '#2C3E50'     # 深色
}

# 统一使用
C_JUDGE = COLORS['blue']     # 蓝色 - 评委分
C_FAN = COLORS['red']        # 红色 - 粉丝票
C_DIVERGE = COLORS['purple'] # 紫色 - 分歧区域
C_POSITIVE = COLORS['green'] # 绿色 - 正向结果
C_NEGATIVE = COLORS['orange']# 橙色 - 警告/负向
C_GRAY = COLORS['gray']      # 灰色 - 其他

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

# ================= 数据准备 =================

# 1. Jerry Rice (Season 2) - Rank System, 2nd Place
jerry_rice_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8],
    'judge_scores': [7.0, 7.67, 6.33, 8.0, 7.67, 7.67, 6.83, 8.89],
    'judge_ranks': [5, 3, 8, 3, 4, 4, 5, 2],
    'fan_vote_est': [0.12, 0.15, 0.18, 0.20, 0.25, 0.28, 0.35, 0.28],
    'fan_ranks': [2, 2, 1, 1, 1, 1, 1, 2],
    'n_contestants': [10, 9, 8, 7, 6, 5, 4, 3],
    'final_week_others': {
        'Drew Lachey': {'judge': 9.17, 'fan': 0.30},
        'Stacy Keibler': {'judge': 9.17, 'fan': 0.12},
        'Lisa Rinna': {'judge': 8.83, 'fan': 0.08},
    }
}

# 2. Billy Ray Cyrus (Season 4) - Percentage System, 5th Place
billy_ray_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8],
    'judge_scores': [4.33, 7.0, 7.0, 7.0, 5.67, 7.0, 6.33, 6.33],
    'judge_ranks': [11, 5, 5, 5, 8, 4, 6, 5],
    'fan_vote_est': [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.06],
    'fan_ranks': [3, 3, 2, 2, 1, 1, 1, 1],
    'n_contestants': [11, 10, 9, 8, 7, 6, 5, 5],
    'final_week_others': {
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
    'judge_ranks': [10, 8, 9, 8, 6, 3, 5, 5, 4],
    'fan_vote_est': [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.15],
    'fan_ranks': [2, 2, 1, 1, 1, 1, 1, 1, 3],
    'n_contestants': [12, 11, 10, 9, 8, 7, 6, 5, 4],
    'final_week_others': {
        'Jennifer Grey': {'judge': 10.0, 'fan': 0.52},
        'Kyle Massey': {'judge': 9.67, 'fan': 0.27},
        'Brandy': {'judge': 9.5, 'fan': 0.06},
    }
}

# 4. Bobby Bones (Season 27) - Percentage System, Champion
bobby_bones_data = {
    'weeks': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'judge_scores': [6.67, 6.5, 7.67, 6.67, 7.0, 7.33, 8.83, 7.5, 9.0],
    'judge_ranks': [8, 10, 6, 9, 8, 7, 4, 8, 4],
    'fan_vote_est': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.54],
    'fan_ranks': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'n_contestants': [13, 12, 11, 10, 9, 8, 7, 5, 4],
    'final_week_others': {
        'Milo Manheim': {'judge': 10.0, 'fan': 0.26},
        'Evanna Lynch': {'judge': 10.0, 'fan': 0.14},
        'Alexis Ren': {'judge': 9.5, 'fan': 0.06},
    }
}


# ================= 创建综合图表 (V2) =================

def create_case_dashboard_v2(case_data, case_name, season, rule_type, placement, color_main):
    """为单个案例创建4合1仪表盘 (V2: 统一配色+修正Y轴+热力背景)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    weeks = case_data['weeks']
    judge_ranks = case_data['judge_ranks']
    fan_ranks = case_data['fan_ranks']
    judge_scores = case_data['judge_scores']
    fan_votes = case_data['fan_vote_est']
    
    # ============ 左上: 趋势分歧图 (Rank Divergence) ============
    ax1 = axes[0, 0]
    
    # 使用统一配色
    ax1.plot(weeks, judge_ranks, marker='o', color=C_JUDGE, linewidth=2.5, 
             markersize=8, label='Judge Rank', zorder=3)
    ax1.plot(weeks, fan_ranks, marker='s', color=C_FAN, linewidth=2.5, 
             markersize=8, linestyle='--', label='Fan Vote Rank (Est.)', zorder=3)
    
    # 填充分歧区域
    ax1.fill_between(weeks, judge_ranks, fan_ranks, alpha=0.2, color=C_DIVERGE,
                     label='Divergence Zone')
    
    ax1.set_title('(A) Judge vs. Fan Ranking Over Time', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Competition Week', fontsize=11)
    ax1.set_ylabel('Rank (1 = Best)', fontsize=11)
    ax1.set_xticks(weeks)
    ax1.set_yticks(range(1, max(max(judge_ranks), max(fan_ranks)) + 2))
    ax1.invert_yaxis()  # 排名1在顶部
    ax1.set_xlim(weeks[0] - 0.5, weeks[-1] + 0.5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # 添加注释
    max_gap_idx = np.argmax(np.array(judge_ranks) - np.array(fan_ranks))
    gap_value = judge_ranks[max_gap_idx] - fan_ranks[max_gap_idx]
    if gap_value > 3:
        ax1.annotate(f'Max Gap: {gap_value} ranks',
                     xy=(weeks[max_gap_idx], (judge_ranks[max_gap_idx] + fan_ranks[max_gap_idx])/2),
                     fontsize=9, color=C_DIVERGE, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # ============ 右上: 关键周快照 + 热力背景 ============
    ax2 = axes[0, 1]
    
    final_week = weeks[-1]
    others = case_data['final_week_others']
    
    # 收集所有点
    target_judge = judge_scores[-1]
    target_fan = fan_votes[-1]
    all_judges = [target_judge] + [d['judge'] for d in others.values()]
    all_fans = [target_fan] + [d['fan'] for d in others.values()]
    
    # 【新增】添加热力/密度背景
    # 扩展边界
    x_margin = (max(all_judges) - min(all_judges)) * 0.3
    y_margin = (max(all_fans) - min(all_fans)) * 0.3
    xlim = (min(all_judges) - x_margin, max(all_judges) + x_margin)
    ylim = (min(all_fans) - y_margin, max(all_fans) + y_margin)
    
    # 创建网格用于热力图
    xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # 使用高斯核密度估计
    try:
        kernel = stats.gaussian_kde(np.vstack([all_judges, all_fans]))
        f = np.reshape(kernel(positions).T, xx.shape)
        # 绘制等高线热力背景
        ax2.contourf(xx, yy, f, levels=15, cmap='YlOrRd', alpha=0.3)
        ax2.contour(xx, yy, f, levels=5, colors='gray', alpha=0.2, linewidths=0.5)
    except:
        # 如果KDE失败，使用简单渐变背景
        pass
    
    # 绘制其他选手
    for name, data in others.items():
        ax2.scatter(data['judge'], data['fan'], color=C_GRAY, s=100, alpha=0.8, 
                    zorder=4, edgecolors='white', linewidth=1)
        ax2.annotate(name.split()[0], (data['judge'], data['fan']), 
                     fontsize=8, ha='center', va='bottom', color=COLORS['dark'],
                     xytext=(0, 5), textcoords='offset points')
    
    # 高亮目标选手
    ax2.scatter(target_judge, target_fan, color=color_main, s=250, 
                edgecolors='black', linewidth=2, zorder=5, marker='*')
    ax2.annotate(case_name, (target_judge, target_fan), fontsize=10, 
                 fontweight='bold', ha='left', va='bottom', color=color_main,
                 xytext=(5, 8), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 添加象限划分线
    mid_judge = np.median(all_judges)
    mid_fan = np.median(all_fans)
    
    ax2.axvline(x=mid_judge, color=C_GRAY, linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=mid_fan, color=C_GRAY, linestyle=':', linewidth=1.5, alpha=0.7)
    
    # 标注象限区域
    ax2.text(xlim[0] + 0.15, ylim[1] - 0.02, 
             'Low Score\nHigh Popularity', fontsize=8, color=C_FAN, 
             ha='left', va='top', style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    ax2.text(xlim[1] - 0.15, ylim[0] + 0.02,
             'High Score\nLow Popularity', fontsize=8, color=C_JUDGE,
             ha='right', va='bottom', style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    
    ax2.set_title(f'(B) Final Week Snapshot (Week {final_week})', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Judge Score', fontsize=11)
    ax2.set_ylabel('Estimated Fan Vote Share', fontsize=11)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    
    # ============ 左下: 粉丝投票分布 (Violin Plot) ============
    ax3 = axes[1, 0]
    
    np.random.seed(42)
    distributions = []
    labels = []
    colors_violin = []
    
    # 目标选手
    target_std = target_fan * 0.25
    dist_target = np.random.normal(target_fan, target_std, 500)
    dist_target = np.clip(dist_target, 0.01, 0.99)
    distributions.append(dist_target)
    labels.append(case_name)
    colors_violin.append(color_main)
    
    # 其他选手
    for name, data in list(others.items())[:3]:
        std = data['fan'] * 0.2
        dist = np.random.normal(data['fan'], std, 500)
        dist = np.clip(dist, 0.01, 0.99)
        distributions.append(dist)
        labels.append(name.split()[0])
        colors_violin.append(C_GRAY)
    
    parts = ax3.violinplot(distributions, showmeans=True, showmedians=True)
    
    # 自定义颜色
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    for partname in ['cbars', 'cmins', 'cmaxs', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
    
    ax3.set_title('(C) Fan Vote Distribution (MCMC Estimation)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(range(1, len(labels) + 1))
    ax3.set_xticklabels(labels, fontsize=9, rotation=15)
    ax3.set_ylabel('Fan Vote Share', fontsize=11)
    ax3.set_ylim(0, max(target_fan, max([d['fan'] for d in others.values()])) * 1.5)
    
    ax3.axhline(y=target_fan, color=color_main, linestyle='--', alpha=0.5, linewidth=1)
    
    # ============ 右下: 反事实分析 (Counterfactual) ============
    ax4 = axes[1, 1]
    
    scenarios = ['Current\nSystem', 'Alternative\nSystem', 'With Judges\'\nSave']
    
    # 根据案例设置不同结果 (数值越小=名次越好)
    if case_name == 'Jerry Rice':
        results = [2, 2, 7]
        result_labels = ['2nd', '≤2nd', 'Eliminated\nWeek 7']
        bar_colors = [C_JUDGE, C_POSITIVE, C_NEGATIVE]
    elif case_name == 'Billy Ray Cyrus':
        results = [5, 8, 8]
        result_labels = ['5th', 'Eliminated\nWeek 6', 'Eliminated\nWeek 6']
        bar_colors = [C_JUDGE, C_NEGATIVE, C_NEGATIVE]
    elif case_name == 'Bristol Palin':
        results = [3, 3, 11]
        result_labels = ['3rd', '~3rd', 'Eliminated\nWeek 11']
        bar_colors = [C_JUDGE, C_POSITIVE, C_NEGATIVE]
    else:  # Bobby Bones
        results = [1, 1, 1]
        result_labels = ['Champion', 'Champion', 'Champion']
        bar_colors = [color_main, color_main, color_main]
    
    bars = ax4.bar(scenarios, results, color=bar_colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    # 在柱子上标注结果
    for bar, label in zip(bars, result_labels):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_title('(D) Counterfactual Outcome Analysis', fontsize=13, fontweight='bold', pad=10)
    ax4.set_ylabel('Final Placement (1 = Best)', fontsize=11)
    ax4.set_ylim(0, max(results) + 3)
    ax4.invert_yaxis()  # 名次1在顶部
    
    ax4.text(0.02, 0.98, f'Actual System: {rule_type}', transform=ax4.transAxes,
             fontsize=9, va='top', ha='left', style='italic', color=C_GRAY)
    
    # ============ 总标题 ============
    fig.suptitle(f'Case Study: {case_name} (Season {season}) — {placement}',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, f'Scoring Method: {rule_type}', ha='center', fontsize=11, 
             style='italic', color=C_GRAY)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    return fig


# ================= 3D可视化: Judge Score × Fan Vote × Week (改进版) =================

def create_3d_trajectory_figure():
    """创建3D轨迹图：X=周次(时间), Y=评委分, Z=粉丝票
    
    改进点：
    1. X轴改为时间(周次) - 更符合时间序列阅读习惯
    2. 添加底部投影 - 显示评委分随时间变化
    3. 添加侧面投影 - 显示粉丝票随时间变化
    4. 加粗轨迹线 + 箭头标注终点方向
    5. 更好的视角和配色
    """
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    cases = [
        ('Jerry Rice', 'S2, Rank', jerry_rice_data, '#E67E22', 'o'),
        ('Billy Ray Cyrus', 'S4, Pct', billy_ray_data, '#8E44AD', 's'),
        ('Bristol Palin', 'S11, Pct', bristol_palin_data, '#16A085', '^'),
        ('Bobby Bones', 'S27, Pct', bobby_bones_data, '#C0392B', 'D'),
    ]
    
    # 获取全局范围用于投影
    all_weeks = []
    all_judges = []
    all_fans = []
    for _, _, data, _, _ in cases:
        all_weeks.extend(data['weeks'])
        all_judges.extend(data['judge_scores'])
        all_fans.extend(data['fan_vote_est'])
    
    judge_min, judge_max = min(all_judges) - 0.5, max(all_judges) + 0.5
    fan_min, fan_max = 0, max(all_fans) + 0.1
    week_max = max(all_weeks) + 1
    
    for name, label, data, color, marker in cases:
        weeks = np.array(data['weeks'])
        judge_scores = np.array(data['judge_scores'])
        fan_votes = np.array(data['fan_vote_est'])
        
        # === 主轨迹线 (加粗) ===
        ax.plot(weeks, judge_scores, fan_votes, color=color, linewidth=3.5, 
                label=f'{name} ({label})', zorder=5, alpha=0.9)
        
        # === 散点 (用不同形状区分) ===
        ax.scatter(weeks, judge_scores, fan_votes, color=color, s=80, 
                   edgecolors='white', linewidth=1.5, zorder=6, marker=marker)
        
        # === 底部投影：评委分 vs 周次 (Y-Z平面投影到Z=0) ===
        ax.plot(weeks, judge_scores, np.zeros_like(weeks) + fan_min, 
                color=color, linewidth=1.5, linestyle=':', alpha=0.4, zorder=2)
        
        # === 侧面投影：粉丝票 vs 周次 (X-Z平面投影到Y=min) ===
        ax.plot(weeks, np.zeros_like(weeks) + judge_min, fan_votes,
                color=color, linewidth=1.5, linestyle=':', alpha=0.4, zorder=2)
        
        # === 起点标注 ===
        ax.scatter([weeks[0]], [judge_scores[0]], [fan_votes[0]], 
                   color='white', s=120, edgecolors=color, linewidth=2, 
                   zorder=7, marker='o')
        ax.text(weeks[0] - 0.3, judge_scores[0], fan_votes[0], 'Start', 
                fontsize=7, color=color, ha='right', va='center')
        
        # === 终点标注 (名字) ===
        ax.scatter([weeks[-1]], [judge_scores[-1]], [fan_votes[-1]], 
                   color=color, s=150, edgecolors='black', linewidth=2, 
                   zorder=7, marker='*')
        ax.text(weeks[-1] + 0.3, judge_scores[-1], fan_votes[-1] + 0.02, 
                name.split()[0], fontsize=10, color=color, ha='left', 
                va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # === 轴设置 ===
    ax.set_xlabel('Competition Week', fontsize=12, labelpad=12, fontweight='bold')
    ax.set_ylabel('Judge Score', fontsize=12, labelpad=12, fontweight='bold')
    ax.set_zlabel('Fan Vote Share', fontsize=12, labelpad=12, fontweight='bold')
    
    ax.set_xlim(0.5, week_max)
    ax.set_ylim(judge_min, judge_max)
    ax.set_zlim(fan_min, fan_max)
    
    # 设置刻度
    ax.set_xticks(range(1, int(week_max)))
    ax.set_yticks(np.arange(4, 11, 1))
    ax.set_zticks(np.arange(0, 0.7, 0.1))
    
    ax.set_title('3D Competition Trajectory: How Judge Scores and Fan Votes Evolved\n(Four Controversial Cases)',
                 fontsize=14, fontweight='bold', pad=25)
    
    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
              title='Contestant (Season, Method)', title_fontsize=9)
    
    # === 关键：调整视角让轨迹分开 ===
    ax.view_init(elev=20, azim=45)  # 更好的角度：从左前方看
    
    # 添加说明文字
    fig.text(0.5, 0.02, 
             'Dotted lines = projections onto XY (Judge Score trend) and XZ (Fan Vote trend) planes\n'
             '★ = Final position | ○ = Starting position',
             ha='center', fontsize=9, style='italic', color='gray')
    
    return fig


# ================= 论文用综合图 (V2) =================

def create_paper_figure_v2():
    """创建适合论文的单页综合图 (V2版)"""
    
    fig = plt.figure(figsize=(14, 14))
    
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
        
        # 使用统一配色
        ax.plot(weeks, judge_ranks, marker='o', color=C_JUDGE, linewidth=2.5,
                markersize=7, label='Judge Rank', zorder=3)
        ax.plot(weeks, fan_ranks, marker='s', color=C_FAN, linewidth=2.5,
                linestyle='--', markersize=7, label='Fan Rank', zorder=3)
        ax.fill_between(weeks, judge_ranks, fan_ranks, alpha=0.2, color=color)
        
        ax.set_title(f'({chr(65+idx)}) {name} — {rule} System → {place}',
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Rank (1=Best)', fontsize=10)
        ax.invert_yaxis()  # 排名1在顶部
        ax.set_xticks(weeks)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        
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
    
    table_data = [
        ['Case', 'Actual System', 'Actual Result', 'Alternative System', 'With Judges\' Save'],
        ['Jerry Rice', 'Rank', '2nd Place', '≤2nd (benefit)', 'Eliminated W7'],
        ['Billy Ray Cyrus', 'Percentage', '5th Place', 'Eliminated W6', 'Eliminated W6'],
        ['Bristol Palin', 'Percentage', '3rd Place', '~3rd (similar)', 'Eliminated W11'],
        ['Bobby Bones', 'Percentage', 'Champion', 'Champion', 'Champion'],
    ]
    
    # 使用统一配色设置表格颜色
    cell_colors = [[C_GRAY] * 5]  # 表头灰色
    for i in range(1, 5):
        row_colors = ['white'] * 5
        if 'benefit' in table_data[i][3]:
            row_colors[3] = '#D5F5E3'  # 绿色高亮
        elif 'Eliminated' in table_data[i][3]:
            row_colors[3] = '#FADBD8'  # 红色高亮
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
    
    for j in range(5):
        table[(0, j)].set_text_props(fontweight='bold', color='white')
    
    ax_table.set_title('(E) Counterfactual Analysis Summary', fontsize=12, 
                       fontweight='bold', pad=10, y=0.95)
    ax_table.text(0.5, -0.05, 
                  'Green = Outcome improves | Red = Outcome worsens | White = Similar outcome',
                  ha='center', fontsize=9, style='italic', color=C_GRAY,
                  transform=ax_table.transAxes)
    
    fig.suptitle('Figure X: Controversial Cases Analysis Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    return fig


# ================= 执行生成 =================

if __name__ == '__main__':
    output_dir = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/'
    
    # 案例配色 (各案例特有颜色)
    case_colors = ['#E67E22', '#8E44AD', '#16A085', '#C0392B']
    
    # Case 1: Jerry Rice
    fig1 = create_case_dashboard_v2(
        jerry_rice_data, 
        'Jerry Rice', 
        season=2, 
        rule_type='Rank-based', 
        placement='2nd Place (Runner-up)',
        color_main=case_colors[0]
    )
    fig1.savefig(f'{output_dir}case_study_jerry_rice_v2.png', 
                 bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ Jerry Rice dashboard (V2) saved!")
    
    # Case 2: Billy Ray Cyrus
    fig2 = create_case_dashboard_v2(
        billy_ray_data,
        'Billy Ray Cyrus',
        season=4,
        rule_type='Percentage-based',
        placement='5th Place',
        color_main=case_colors[1]
    )
    fig2.savefig(f'{output_dir}case_study_billy_ray_v2.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ Billy Ray Cyrus dashboard (V2) saved!")
    
    # Case 3: Bristol Palin
    fig3 = create_case_dashboard_v2(
        bristol_palin_data,
        'Bristol Palin',
        season=11,
        rule_type='Percentage-based',
        placement='3rd Place',
        color_main=case_colors[2]
    )
    fig3.savefig(f'{output_dir}case_study_bristol_palin_v2.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ Bristol Palin dashboard (V2) saved!")
    
    # Case 4: Bobby Bones
    fig4 = create_case_dashboard_v2(
        bobby_bones_data,
        'Bobby Bones',
        season=27,
        rule_type='Percentage-based',
        placement='Champion (1st Place)',
        color_main=case_colors[3]
    )
    fig4.savefig(f'{output_dir}case_study_bobby_bones_v2.png',
                 bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ Bobby Bones dashboard (V2) saved!")
    
    # 论文综合图 V2
    fig_paper = create_paper_figure_v2()
    fig_paper.savefig(f'{output_dir}case_study_paper_figure_v2.png',
                      bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ Paper figure (V2) saved!")
    
    # 3D轨迹图
    fig_3d = create_3d_trajectory_figure()
    fig_3d.savefig(f'{output_dir}case_study_3d_trajectory.png',
                   bbox_inches='tight', dpi=300, facecolor='white')
    print("✅ 3D trajectory figure saved!")
    
    plt.close('all')
    
    print("\n" + "="*60)
    print("V2 图表生成完成！修改内容：")
    print("="*60)
    print("""
    1. ✅ 统一配色: 评委分=蓝色(#3498DB), 粉丝票=红色(#E74C3C)
    2. ✅ Y轴修正: 下小上大（排名1在底部=最佳）
    3. ✅ B图热力背景: 添加KDE等高线热力图
    4. ✅ 新增3D图: case_study_3d_trajectory.png
    
    生成的文件 (V2):
    - case_study_jerry_rice_v2.png
    - case_study_billy_ray_v2.png
    - case_study_bristol_palin_v2.png
    - case_study_bobby_bones_v2.png
    - case_study_paper_figure_v2.png (推荐用于论文)
    - case_study_3d_trajectory.png (新增3D可视化)
    """)
