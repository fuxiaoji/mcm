% PCA主成分分析法案例说明：
% 本案例使用主成分分析(PCA)对高维数据进行降维。PCA是一种常用的降维技术，
% 通过线性变换将高维数据映射到低维空间，同时保留数据中最重要的信息。
% 示例中使用鸢尾花数据集(4维特征)，通过PCA降维到2维空间进行可视化。

% 加载数据集
load fisheriris;  % 加载鸢尾花数据集
X = meas;         % 4维特征数据
y = species;      % 标签
feature_names = {'花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'};
target_names = unique(y);

% 数据标准化（PCA对数据尺度敏感，通常需要标准化）
X_scaled = zscore(X);  % 标准化：均值为0，标准差为1

% 应用PCA
[coeff, score, latent, ~, explained] = pca(X_scaled);

% 选择前2个主成分
X_pca = score(:, 1:2);

% 输出PCA结果分析
fprintf('PCA主成分分析结果：\n');
fprintf('原始数据维度: %d 维\n', size(X, 2));
fprintf('降维后数据维度: 2 维\n');
fprintf('\n主成分解释方差比例:\n');
for i = 1:2
    fprintf('主成分 %d: %.4f (%.2f%%)\n', i, explained(i)/100, explained(i));
end
fprintf('\n累计解释方差比例: %.4f (%.2f%%)\n', sum(explained(1:2))/100, sum(explained(1:2)));

fprintf('\n主成分载荷矩阵（表示原始特征与主成分的相关性）:\n');
loadings = coeff(:, 1:2) * diag(sqrt(latent(1:2)));
disp(array2table(loadings, 'RowNames', feature_names, 'VariableNames', {'主成分1', '主成分2'}));

% 可视化PCA降维结果
figure;
colors = lines(3);  % 三种颜色
markers = {'o', 's', 'd'};  % 三种标记

for i = 1:length(target_names)
    idx = strcmp(y, target_names{i});
    plot(X_pca(idx, 1), X_pca(idx, 2), 'o', 'MarkerFaceColor', colors(i,:), ...
         'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'DisplayName', target_names{i});
    hold on;
end

xlabel(sprintf('主成分 1 (%.2f%%)', explained(1)));
ylabel(sprintf('主成分 2 (%.2f%%)', explained(2)));
title('PCA降维：鸢尾花数据集(4维→2维)');
legend('Location', 'best');
grid on;
box on;
hold off;

% 绘制解释方差比例条形图
figure;
bar(explained(1:4), 'FaceColor', [0.5 0.8 1]);
hold on;
plot(cumsum(explained(1:4)), 'r-', 'LineWidth', 2);
xlabel('主成分数量');
ylabel('解释方差比例 (%)');
title('PCA解释方差比例');
legend('单个主成分解释方差', '累计解释方差', 'Location', 'best');
grid on;
box on;
hold off;
    