% UMAP降维法案例说明：
% 本案例使用UMAP(均匀流形近似和投影)算法对高维数据进行降维。
% UMAP是一种现代的非线性降维方法，相比t-SNE通常保留更多的全局结构，计算速度也更快。
% 示例中使用手写数字数据集(64维)，通过UMAP降维到2维空间进行可视化。

% 检查并安装必要的工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox来运行UMAP');
end

% 加载数据集（使用MATLAB内置数据集）
load('digits.mat');  % 手写数字数据集(64维)
X = digits.images;
X = reshape(X, size(X,1)*size(X,2), size(X,3))';  % 转换为特征矩阵
y = digits.labels + 1;  % 标签从1开始

% 只使用部分样本加快计算
n_samples = min(500, size(X,1));
X_sample = X(1:n_samples, :);
y_sample = y(1:n_samples);
original_dim = size(X_sample, 2);

% 数据标准化
X_scaled = zscore(X_sample);  % 标准化：均值为0，标准差为1

% 应用UMAP降维到2维
fprintf('运行UMAP降维...\n');
params.NumNeighbors = 15;    % 近邻数量
params.MinDistance = 0.1;    % 最小距离
params.Seed = 42;            % 随机种子，确保结果可复现
X_umap = umap(X_scaled, params);

% 输出UMAP结果信息
fprintf('\nUMAP降维结果：\n');
fprintf('原始数据维度: %d 维\n', original_dim);
fprintf('降维后数据维度: %d 维\n', size(X_umap, 2));
fprintf('使用参数: n_neighbors=%d, min_dist=%.1f\n', ...
        params.NumNeighbors, params.MinDistance);

% 可视化UMAP降维结果
figure('Position', [100 100 800 700]);
colors = lines(10);  % 10种颜色
markers = {'o', 's', '^', 'D', '*', 'P', 'X', 'd', 'p', 'h'};  % 10种标记

for digit = 1:10
    idx = y_sample == digit;
    plot(X_umap(idx, 1), X_umap(idx, 2), 'Marker', markers{digit}, ...
         'MarkerFaceColor', colors(digit,:), 'MarkerEdgeColor', 'k', ...
         'MarkerSize', 8, 'LineStyle', 'none', 'DisplayName', num2str(digit-1));
    hold on;
end

legend('Location', 'eastoutside', 'Title', '数字');
title(sprintf('UMAP降维：手写数字数据集(%d维→2维)', original_dim));
xlabel('UMAP特征1');
ylabel('UMAP特征2');
grid on;
box on;
hold off;
    