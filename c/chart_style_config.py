"""
MCM Problem C - 统一配色方案
====================================
只提供颜色定义，不改变原有图表样式

使用方法:
    from chart_style_config import COLORS, PALETTE
    
    # 在绑图时使用统一颜色
    plt.bar(x, y, color=COLORS['blue'])
"""

# ==========================================
# 统一配色方案 - 学术论文风格
# ==========================================

COLORS = {
    # 主色系 - 用于数据系列
    'blue': '#3498DB',          # 天蓝 - 主数据/评委/Percentage方法
    'red': '#E74C3C',           # 珊瑚红 - 对比数据/粉丝/Rank方法
    'green': '#2ECC71',         # 翠绿 - 正面/安全/有效
    'orange': '#F39C12',        # 橙黄 - 警告/冠军/强调
    'purple': '#9B59B6',        # 紫色 - 综合/第三维度
    'cyan': '#1ABC9C',          # 青色 - 补充色
    
    # 语义色
    'positive': '#27AE60',      # 深绿 - 正面/验证通过
    'negative': '#C0392B',      # 深红 - 负面/争议
    'neutral': '#7F8C8D',       # 灰色 - 中性
    
    # 辅助色
    'dark': '#2C3E50',          # 深蓝灰 - 文字/标题
    'light_gray': '#BDC3C7',    # 浅灰 - 边框
    'grid': '#E5E8E8',          # 网格线
}

# 多数据系列调色板
PALETTE = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

# 特定场景配色
METHOD_COLORS = {
    'percentage': '#3498DB',    # 蓝色 - Percentage方法
    'rank': '#E74C3C',          # 红色 - Rank方法
}

ANALYSIS_COLORS = {
    'judge': '#3498DB',         # 蓝色 - 评委
    'fan': '#E74C3C',           # 红色 - 粉丝
    'combined': '#9B59B6',      # 紫色 - 综合
}

RESULT_COLORS = {
    'valid': '#27AE60',         # 绿色 - 有效/验证
    'controversial': '#C0392B', # 红色 - 争议
    'neutral': '#7F8C8D',       # 灰色 - 中性
}

# 选手配色（问题5案例分析用）
CONTESTANT_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']  # Bobby, Milo, Evanna, Alexis


if __name__ == '__main__':
    print("MCM Problem C 统一配色方案")
    print("="*40)
    print("\n主色系:")
    for name, color in COLORS.items():
        print(f"  {name}: {color}")
    print(f"\n调色板: {PALETTE}")
