"""
PCA 3D 散点图 + 置信椭圆 (V2)
=============================
按赛季时期分组，更清晰的可视化效果

改进点：
1. 按赛季时期分组（3组，差异更明显）
2. 更少的散点、更大的椭圆
3. 更好的视角和配色
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# 配色方案 - 按赛季时期（类似参考图的年份分组）
COLORS_SEASON = {
    'Early (S1-10)': '#1E88E5',      # 蓝色
    'Middle (S11-20)': '#E53935',    # 红色
    'Recent (S21-33)': '#43A047',    # 绿色
}

# ================= 数据加载与处理 =================

def load_and_prepare_data():
    """加载选手汇总表并准备PCA所需数据"""
    
    df = pd.read_csv('/Users/Zhuanz1/Desktop/mcm/c/问题1_完整分析/问题1_选手汇总表.csv')
    
    # 选择用于PCA的数值特征
    pca_features = [
        'celebrity_age',        # 年龄
        'weeks_survived',       # 存活周数
        'season_avg_score',     # 平均评委分
        'fan_vote_estimate',    # 粉丝票估计
        'placement',            # 最终名次
    ]
    
    # 筛选有完整数据的行
    df_clean = df.dropna(subset=pca_features + ['season'])
    
    # 按赛季时期分组
    def get_season_era(season):
        if season <= 10:
            return 'Early (S1-10)'
        elif season <= 20:
            return 'Middle (S11-20)'
        else:
            return 'Recent (S21-33)'
    
    df_clean['season_era'] = df_clean['season'].apply(get_season_era)
    
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
    
    print("PCA方差解释比例:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.2%}")
    print(f"  累计: {sum(pca.explained_variance_ratio_):.2%}")
    
    return X_pca, pca


def draw_ellipsoid_clean(ax, center, cov, color, alpha=0.2):
    """绘制清晰的置信椭球"""
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.1)
    
    # 生成椭球面
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    
    # 95%置信区间 (chi2 df=3)
    n_std = np.sqrt(7.815)
    radii = n_std * np.sqrt(eigenvalues)
    
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # 旋转
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i,j], y[i,j], z[i,j]])
            rotated = eigenvectors @ point + center
            x[i,j], y[i,j], z[i,j] = rotated
    
    # 绘制半透明表面
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, 
                    shade=True, antialiased=True)
    # 绘制轮廓线
    ax.plot_wireframe(x, y, z, color=color, alpha=0.4, linewidth=0.5,
                      rstride=8, cstride=8)


def create_pca_3d_figure(df_clean, X_pca, pca):
    """创建3D PCA散点图（按赛季时期分组）"""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # 按赛季时期分组绘制
    eras = ['Early (S1-10)', 'Middle (S11-20)', 'Recent (S21-33)']
    markers = ['o', 's', '^']
    
    for era, marker in zip(eras, markers):
        mask = (df_clean['season_era'] == era).values
        points = X_pca[mask]
        
        if len(points) < 5:
            continue
        
        color = COLORS_SEASON[era]
        
        # 绘制散点（更小更清晰）
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=25, alpha=0.7, edgecolors='white',
                   linewidth=0.3, marker=marker,
                   label=f'{era} (n={len(points)})')
        
        # 计算并绘制置信椭球
        center = np.mean(points, axis=0)
        cov = np.cov(points.T)
        
        try:
            draw_ellipsoid_clean(ax, center, cov, color, alpha=0.15)
            print(f"   ✅ {era} 椭球绘制成功")
        except Exception as e:
            print(f"   ❌ {era} 椭球绘制失败: {e}")
    
    # 轴标签
    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=11, labelpad=8)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=11, labelpad=8)
    ax.set_zlabel(f'PC3 ({var_ratio[2]:.1%})', fontsize=11, labelpad=8)
    
    ax.set_title('Principal Component Analysis: Contestant Profiles by Season Era',
                 fontsize=13, fontweight='bold', pad=15)
    
    # 图例
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
              title='Season Era', title_fontsize=10)
    
    # 调整视角
    ax.view_init(elev=25, azim=30)
    
    # 底部说明
    fig.text(0.5, 0.02,
             '95% Confidence Ellipsoids for each season era',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig


def create_pca_2d_supplementary(df_clean, X_pca, pca):
    """创建2D PCA投影图"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    eras = ['Early (S1-10)', 'Middle (S11-20)', 'Recent (S21-33)']
    markers = ['o', 's', '^']
    
    for era, marker in zip(eras, markers):
        mask = (df_clean['season_era'] == era).values
        points = X_pca[mask]
        
        if len(points) < 5:
            continue
        
        color = COLORS_SEASON[era]
        
        ax.scatter(points[:, 0], points[:, 1],
                   c=color, s=35, alpha=0.6, edgecolors='white',
                   linewidth=0.3, marker=marker,
                   label=f'{era} (n={len(points)})')
        
        # 95%置信椭圆
        center = np.mean(points[:, :2], axis=0)
        cov = np.cov(points[:, :2].T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        width, height = 2 * np.sqrt(5.991) * np.sqrt(eigenvalues)
        
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(center, width, height, angle=angle,
                          facecolor=color, alpha=0.15, 
                          edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
    
    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=12)
    ax.set_title('PCA: Contestant Profiles by Season Era (2D Projection)',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    ax.text(0.02, 0.02, 'Ellipses: 95% Confidence Regions',
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    return fig


# ================= 主程序 =================

if __name__ == '__main__':
    print("="*60)
    print("PCA 3D 可视化 V2：选手特征按赛季时期分组")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    df_clean, X_scaled, features = load_and_prepare_data()
    print(f"   有效样本数: {len(df_clean)}")
    print(f"   使用特征: {features}")
    
    # 统计各时期样本数
    print("\n   各赛季时期样本数:")
    for era, count in df_clean['season_era'].value_counts().items():
        print(f"     {era}: {count}")
    
    # 2. PCA降维
    print("\n2. 执行PCA...")
    X_pca, pca = perform_pca(X_scaled, n_components=3)
    
    # 3. 生成3D图
    print("\n3. 生成3D PCA图...")
    fig_3d = create_pca_3d_figure(df_clean, X_pca, pca)
    output_path_3d = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/pca_3d_season_era.png'
    fig_3d.savefig(output_path_3d, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"   ✅ 3D图已保存: {output_path_3d}")
    
    # 4. 生成2D补充图
    print("\n4. 生成2D PCA图...")
    fig_2d = create_pca_2d_supplementary(df_clean, X_pca, pca)
    output_path_2d = '/Users/Zhuanz1/Desktop/mcm/c/论文图表/pca_2d_season_era.png'
    fig_2d.savefig(output_path_2d, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"   ✅ 2D图已保存: {output_path_2d}")
    
    plt.close('all')
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
