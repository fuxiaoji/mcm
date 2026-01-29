% 层次聚类算法案例说明：
% 层次聚类是一种无监督聚类方法，通过构建聚类树（dendrogram）来展示数据的层次聚类结构。
% 主要分为两种：
% 1. 凝聚式（自底向上）：从单个样本开始，逐步合并最相似的簇
% 2. 分裂式（自顶向下）：从所有样本为一个簇开始，逐步分裂为更小的簇
%
% 本案例使用葡萄酒数据集，采用凝聚式层次聚类，通过不同的距离度量和链接方法进行聚类，
% 并绘制聚类树和聚类结果可视化。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行层次聚类');
end

% 加载数据集
load wine;
X = meas;  % 特征数据
y_true = species;  % 真实标签（仅用于对比）

% 数据标准化
X_scaled = zscore(X);  % 标准化：均值为0，标准差为1

% 执行层次聚类（使用Ward链接法，基于欧氏距离）
% 常用链接方法：'ward'（最小化簇内方差）、'average'（平均距离）、'complete'（最大距离）
Z = linkage(X_scaled, 'ward', 'euclidean');

% 输出聚类树信息
fprintf('层次聚类树（前10个合并步骤）：\n');
disp(array2table(round(Z(1:10,:), 2), ...
    'VariableNames', {'簇1', '簇2', '距离', '样本数'}));

% 绘制聚类树（dendrogram）
figure('Position', [100 100 1200 600]);
dendrogram(Z, ...
    'Truncate', 'lastp', ...  % 只显示最后p个合并
    'P', 20, ...              % 显示的合并数
    'LeafRotation', 90, ...   % 叶子节点旋转角度
    'FontSize', 10, ...       % 字体大小
    'ColorThreshold', 5);     % 颜色阈值
title('层次聚类树（Ward链接法）');
xlabel('样本或簇');
ylabel('距离');
grid on;

% 根据聚类树确定聚类数量并获取聚类结果
k = 3;  % 已知葡萄酒数据集有3类
idx = cluster(Z, 'MaxClust', k);  % 划分为k个簇
idx = idx - 1;  % 调整标签从0开始

% 输出聚类结果
fprintf('\n层次聚类结果 (k=%d):\n', k);
for i = 0:k-1
    fprintf('簇 %d: %d 个样本\n', i, sum(idx == i));
end

% 使用PCA降维到2维以便可视化
[coeff, score, latent] = pca(X_scaled);
X_pca = score(:, 1:2);  % 取前两个主成分

% 可视化聚类结果
figure('Position', [100 100 1000 500]);

% 1. 层次聚类结果
subplot(1, 2, 1);
colors = lines(k);
for i = 0:k-1
    plot(X_pca(idx == i, 1), X_pca(idx == i, 2), 'o', ...
        'MarkerFaceColor', colors(i+1,:), 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'DisplayName', sprintf('簇 %d', i));
    hold on;
end
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
title(sprintf('层次聚类结果 (k=%d)', k));
legend('Location', 'best');
grid on;
hold off;

% 2. 与真实标签对比
subplot(1, 2, 2);
unique_species = unique(y_true);
for i = 1:length(unique_species)
    idx_true = strcmp(y_true, unique_species{i});
    plot(X_pca(idx_true, 1), X_pca(idx_true, 2), 'o', ...
        'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'DisplayName', unique_species{i});
    hold on;
end
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
title('真实标签分布');
legend('Location', 'best');
grid on;
hold off;

% 对比不同链接方法的聚类树
figure('Position', [100 200 1500 500]);
methods = {'ward', 'average', 'complete'};
for i = 1:length(methods)
    subplot(1, 3, i);
    Z_method = linkage(X_scaled, methods{i}, 'euclidean');
    dendrogram(Z_method, ...
        'Truncate', 'lastp', ...
        'P', 10, ...
        'LeafRotation', 90, ...
        'FontSize', 8);
    title(sprintf('链接方法: %s', methods{i}));
    xlabel('样本或簇');
    ylabel('距离');
    grid on;
end
    