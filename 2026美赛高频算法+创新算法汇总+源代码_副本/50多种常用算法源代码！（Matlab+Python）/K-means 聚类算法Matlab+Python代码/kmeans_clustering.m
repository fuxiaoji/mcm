% K-means聚类算法案例说明：
% K-means是一种常用的无监督聚类算法，通过迭代方式将数据集划分为k个不同的簇。
% 算法步骤：
% 1. 随机选择k个初始质心
% 2. 将每个样本分配到最近的质心所在的簇
% 3. 重新计算每个簇的质心（平均值）
% 4. 重复步骤2-3，直到质心不再显著变化或达到最大迭代次数
%
% 本案例使用鸢尾花数据集（4维特征），通过K-means聚为3类，
% 并将结果可视化展示（使用PCA降维到2维以便可视化）。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行K-means聚类');
end

% 加载数据集
load fisheriris;
X = meas;  % 特征数据（4维）
y_true = species;  % 真实标签（仅用于对比）
feature_names = {'花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'};

% 数据标准化
X_scaled = zscore(X);  % 标准化：均值为0，标准差为1

% 设置聚类数量（已知鸢尾花有3个品种）
k = 3;

% 执行K-means聚类
[idx, C, sumd, D] = kmeans(X_scaled, k, ...
    'Options', statset('MaxIter', 300), ...  % 最大迭代次数
    'Replicates', 10, ...                    % 多次初始化取最优
    'Start', 'plus', ...                     % 使用k-means++初始化
    'RandomSeed', 42);                       % 随机种子，确保可复现

% 输出聚类结果信息
fprintf('K-means聚类结果 (k=%d):\n', k);
fprintf('最终惯性值(簇内平方和): %.4f\n', sum(sumd));
fprintf('\n每个簇的样本数量:\n');
for i = 1:k
    fprintf('簇 %d: %d 个样本\n', i, sum(idx == i));
end

% 使用PCA降维到2维以便可视化
[coeff, score, latent] = pca(X_scaled);
X_pca = score(:, 1:2);  % 取前两个主成分
C_pca = (C - mean(X_scaled)) * coeff(:, 1:2);  % 转换质心到PCA空间

% 可视化聚类结果
figure('Position', [100 100 1000 500]);

% 1. 聚类结果可视化
subplot(1, 2, 1);
colors = lines(k);
for i = 1:k
    plot(X_pca(idx == i, 1), X_pca(idx == i, 2), 'o', ...
        'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'DisplayName', sprintf('簇 %d', i));
    hold on;
end
% 绘制质心
plot(C_pca(:, 1), C_pca(:, 2), 'X', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', 'LineWidth', 2, ...
    'DisplayName', '质心');
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
title(sprintf('K-means聚类结果 (k=%d)', k));
legend('Location', 'best');
grid on;
hold off;

% 2. 与真实标签对比（仅用于演示）
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

% 尝试不同k值，通过肘部法确定最佳k值
inertias = zeros(1, 10);
for k_test = 1:10
    [~, ~, sumd_test] = kmeans(X_scaled, k_test, ...
        'Replicates', 5, 'RandomSeed', 42);
    inertias(k_test) = sum(sumd_test);
end

% 绘制肘部图
figure('Position', [100 200 800 500]);
plot(1:10, inertias, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('k值（聚类数量）');
ylabel('惯性值（簇内平方和）');
title('肘部法确定最佳k值');
grid on;
    