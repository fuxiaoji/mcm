"""
PCA 3D 散点图 + 置信椭圆
========================
基于选手汇总表数据，按行业分组进行PCA降维可视化

生成类似论文中Figure 6的效果：
- 3D散点图展示PC1, PC2, PC3
- 不同颜色代表不同行业
- 每个行业添加95%置信椭圆
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================= 全局风格设置 =================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

# 配色方案 - 与项目统一
COLORS = {
    'Actor/Actress': '#3498DB',      # 蓝色 - 演员
    'Athlete': '#E74C3C',            # 红色 - 运动员
    'TV Personality': '#2ECC71',     # 绿色 - 电视人物
    'Singer/Rapper': '#F39C12',      # 橙色 - 歌手
    'Model': '#9B59B6',              # 紫色 - 模特
    'Other': '#95A5A6',              # 灰色 - 其他
}

# ================= 数据加载与处理 =================

def load_and_prepare_data():
    """加载选手汇总表并准备PCA所需数据"""
    
    # 加载数据
    df = pd.read_csv('/Users/Zhuanz1/Desktop/mcm/c/问题1_完整分析/问题1_选手汇总表.csv')
    
    # 选择用于PCA的数值特征
    pca_features = [
        'celebrity_age',        # 年龄
        'weeks_survived',       # 存活周数
        'season_avg_score',     # 平均评委分
        'season_score_std',     # 评委分标准差
        'fan_vote_estimate',    # 粉丝票估计
        'fan_certainty',        # 粉丝票确定性
    ]
    
    # 筛选有完整数据的行
    df_clean = df.dropna(subset=pca_features + ['celebrity_industry'])
    
    # 简化行业分类
    def simplify_industry(industry):
        if pd.isna(industry):
            return 'Other'
        industry = str(industry)
        if 'Actor' in industry or 'Actress' in industry:
            return 'Actor/Actress'
        elif 'Athlete' in industry:
            return 'Athlete'
        elif 'TV' in industry:
            return 'TV Personality'
        elif 'Singer' in industry or 'Rapper' in industry:
            return 'Singer/Rapper'
        elif 'Model' in industry:
            return 'Model'
        else:
            return 'Other'
    
    df_clean['industry_group'] = df_clean['celebrity_industry'].apply(simplify_industry)
    
    # 提取特征矩阵
    X = df_clean[pca_features].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled, pca_features


def perform_pca(X_scaled, n_components=3):
    """执行PCA降维"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 打印方差解释比例
    print("PCA方差解释比例:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.2%}")
    print(f"  累计: {sum(pca.explained_variance_ratio_):.2%}")
    
    return X_pca, pca


def draw_ellipsoid(ax, center, cov, color, alpha=0.2, n_std=2.0):
    """在3D图中绘制置信椭球（改进版）"""
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 确保特征值为正
    eigenvalues = np.maximum(eigenvalues, 0.01)
    
    # 生成椭球面上的点（增加分辨率）
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    
    # 标准椭球
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # 缩放
    radii = n_std * np.sqrt(eigenvalues)
    
    # 应用旋转和平移
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i,j] * radii[0], y[i,j] * radii[1], z[i,j] * radii[2]])
            rotated = eigenvectors @ point + center
            x[i,j], y[i,j], z[i,j] = rotated
    
    # 绘制椭球表面（使用wireframe更清晰）
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, 
                    shade=True, antialiased=True)
    # 添加线框轮廓
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha*2, linewidth=0.3, 
                      rstride=5, cstride=5)


def create_pca_3d_figure(df_clean, X_pca, pca):
    """创建3D PCA散点图 + 置信椭球"""
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景透明
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # 按行业分组绘制
    industries = df_clean['industry_group'].unique()
    legend_elements = []
    
    for industry in sorted(industries):
        if industry == 'Other':
            continue  # 跳过"其他"类别以保持图表清晰
            
        mask = df_clean['industry_group'] == industry
        points = X_pca[mask]
        
        if len(points) < 5:  # 样本太少跳过
            continue
        
        color = COLORS.get(industry, '#95A5A6')
        
        # 绘制散点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=50, alpha=0.7, edgecolors='white', 
                   linewidth=0.5, label=f'{industry} (n={len(points)})')
        
        # 计算置信椭球
        center = np.mean(points, axis=0)
        cov = np.cov(points.T)
        
        # 绘制95%置信椭球
        try:
            draw_ellipsoid(ax, center, cov, color, alpha=0.15, n_std=2.0)
            print(f"   ✅ {industry} 椭球绘制成功")
        except Exception as e:
            print(f"   ❌ {industry} 椭球绘制失败: {e}")
        
        # 标注中心点
        ax.scatter([center[0]], [center[1]], [center[2]], 
                   c=color, s=200, marker='X', edgecolors='black', 
                   linewidth=2, zorder=10)
    
    # 轴标签（包含方差解释比例）
    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=12, labelpad=10)
    ax.set_zlabel(f'PC3 ({var_ratio[2]:.1%})', fontsize=12, labelpad=10)
    
    ax.set_title('Principal Component Analysis: Contestant Profiles by Industry\n'
                 '(Features: Age, Weeks Survived, Scores, Fan Votes)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
              title='Celebrity Industry', title_fontsize=10)
    
    # 视角
    ax.view_init(elev=20, azim=45)
    
    # 底部说明
    fig.text(0.5, 0.02,
             'Ellipsoids represent 95% confidence regions for each industry group\n'
             'X markers indicate group centroids',
             ha='center', fontsize=9, style='italic', color='gray')
    
    return fig


def create_pca_2d_supplementary(df_clean, X_pca, pca):
    """创建补充的2D PCA投影图（PC1 vs PC2）"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    industries = df_clean['industry_group'].unique()
    
    for industry in sorted(industries):
        if industry == 'Other':
            continue
            
        mask = df_clean['industry_group'] == industry
        points = X_pca[mask]
        
        if len(points) < 5:
            continue
        
        color = COLORS.get(industry, '#95A5A6')
        
        # 绘制散点
        ax.scatter(points[:, 0], points[:, 1],
                   c=color, s=60, alpha=0.6, edgecolors='white',
                   linewidth=0.5, label=f'{industry} (n={len(points)})')
        
        # 绘制95%置信椭圆
        center = np.mean(points[:, :2], axis=0)
        cov = np.cov(points[:, :2].T)
        
        # 椭圆参数
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        width, height = 2 * 1.96 * np.sqrt(eigenvalues)  # 95% CI
        
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(center, width, height, angle=angle,
                          facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
        
        # 标注中心
        ax.scatter([center[0]], [center[1]], c=color, s=150, marker='X',
                   edgecolors='black', linewidth=2, zorder=10)
    
    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=12)
    ax.set_title('PCA: Contestant Profiles by Industry (2D Projection)',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加说明
    ax.text(0.02, 0.02, 'Ellipses: 95% Confidence Regions',
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig


# ================= 主程序 =================

if __name__ == '__main__':
    print("="*60)
    print("PCA 3D 可视化：选手特征按行业分组")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    df_clean, X_scaled, features = load_and_prepare_data()
    print(f"   有效样本数: {len(df_clean)}")
    print(f"   使用特征: {features}")
    
    # 统计各行业样本数
    print("\n   各行业样本数:")
    for industry, count in df_clean['industry_group'].value_counts().items():
        print(f"     {industry}: {count}")
    
    # 2. PCA降维
    print("\n2. 执行PCA...")
    X_pca, pca = perform_pca(X_scaled, n_components=3)
    
    # 3. 生成3D图
    print("\n3. 生成3D PCA图...")
    fig_3d = create_pca_3d_figure(df_clean, X_pca, pca)
    output_path_3d = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/pca_3d_industry.png'
    fig_3d.savefig(output_path_3d, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"   ✅ 3D图已保存: {output_path_3d}")
    
    # 4. 生成2D补充图
    print("\n4. 生成2D PCA图...")
    fig_2d = create_pca_2d_supplementary(df_clean, X_pca, pca)
    output_path_2d = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/pca_2d_industry.png'
    fig_2d.savefig(output_path_2d, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"   ✅ 2D图已保存: {output_path_2d}")
    
    plt.close('all')
    
    print("\n" + "="*60)
    print("完成！生成的文件:")
    print("="*60)
    print(f"""
    1. pca_3d_industry.png  - 3D PCA散点图+置信椭球
    2. pca_2d_industry.png  - 2D PCA投影图+置信椭圆（补充）
    
    图表说明：
    - 使用6个特征进行PCA: age, weeks_survived, avg_score, 
      score_std, fan_vote_estimate, fan_certainty
    - 按celebrity_industry分组（演员/运动员/歌手/电视人物/模特）
    - 椭球/椭圆表示各组的95%置信区间
    - X标记表示各组中心点
    """)
