% 高斯混合聚类模型（GMM）案例说明：
% 高斯混合模型假设数据是由多个高斯分布的混合生成的，通过期望最大化（EM）算法估计每个高斯分布的参数。
% 与K-means不同，GMM是软聚类方法，每个样本属于每个簇的概率是连续的。
%
% 算法特点：
% 1. 假设数据由k个高斯分布混合而成
% 2. 使用EM算法估计每个高斯分布的均值、协方差和权重
% 3. 可以处理非球形簇和重叠簇
% 4. 输出每个样本属于各个簇的概率
%
% 本案例使用生成的二维模拟数据（3个不同的高斯分布），通过GMM进行聚类，
% 并可视化聚类结果和每个高斯分布的轮廓。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行GMM');
end

% 生成模拟数据（3个不同的高斯分布）
rng(42);  % 设置随机种子，确保结果可复现

% 定义3个高斯分布的参数
means = [0 5; 5 0; 10 5];  % 均值
covariances = {[2 1; 1 2], [3 -1; -1 2], [1 0; 0 3]};  % 协方差矩阵
weights = [0.3, 0.5, 0.2];  % 权重
n_samples = 500;  % 总样本数

% 生成数据
X = zeros(n_samples, 2);
y_true = zeros(n_samples, 1);
for i = 1:n_samples
    % 随机选择一个高斯分布
    cluster = randsample(3, 1, true, weights);
    y_true(i) = cluster;
    % 从选中的高斯分布生成样本
    X(i, :) = mvnrnd(means(cluster, :), covariances{cluster});
end

% 数据标准化
X_scaled = zscore(X);  % 标准化：均值为0，标准差为1

% 构建并训练GMM模型
k = 3;  % 已知有3个簇
gmm = fitgmdist(X_scaled, k, ...
    'CovarianceType', 'full', ...  % 每个分量有自己的完整协方差矩阵
    'MaxIter', 100, ...            % 最大迭代次数
    'Options', statset('Display', 'off'), ...
    'RegularizationValue', 1e-5);  % 正则化，避免奇异矩阵

% 获取聚类结果
[y_pred, probs] = predict(gmm, X_scaled);  % 聚类结果和概率
y_pred = y_pred - 1;  % 调整标签从0开始

% 输出GMM模型参数
fprintf('高斯混合模型（GMM）结果：\n');
fprintf('迭代次数: %d\n', gmm.NumIterations);
fprintf('对数似然值: %.4f\n', gmm.LogLikelihood);

fprintf('\n每个高斯分量的权重:\n');
for i = 1:k
    fprintf('分量 %d: %.4f\n', i-1, gmm.ComponentProportion(i));
end

fprintf('\n每个高斯分量的均值:\n');
for i = 1:k
    fprintf('分量 %d: [%.4f, %.4f]\n', i-1, gmm.Mu(i, 1), gmm.Mu(i, 2));
end

% 可视化聚类结果
figure('Position', [100 100 1000 500]);

% 1. 聚类结果散点图
subplot(1, 2, 1);
colors = lines(k);
for i = 0:k-1
    plot(X_scaled(y_pred == i, 1), X_scaled(y_pred == i, 2), 'o', ...
        'MarkerFaceColor', colors(i+1,:), 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'DisplayName', sprintf('簇 %d', i));
    hold on;
end

% 绘制每个高斯分布的均值
plot(gmm.Mu(:, 1), gmm.Mu(:, 2), 'X', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', 'LineWidth', 2, ...
    'DisplayName', '均值');

title(sprintf('GMM聚类结果 (k=%d)', k));
xlabel('特征1');
ylabel('特征2');
legend('Location', 'best');
grid on;
hold off;

% 2. 绘制高斯分布轮廓（椭圆表示）
subplot(1, 2, 2);
for i = 1:k
    % 绘制样本点
    idx = y_pred == i-1;
    plot(X_scaled(idx, 1), X_scaled(idx, 2), 'o', ...
        'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none', ...
        'MarkerSize', 6, 'AlphaData', 0.3);
    hold on;
    
    % 绘制高斯分布的椭圆轮廓（95%置信区间）
    mu = gmm.Mu(i, :);
    sigma = gmm.Sigma(:,:,i);
    
    % 绘制椭圆
    ezcontour(@(x,y) mvnpdf([x,y], mu, sigma), ...
        [min(X_scaled(:,1))-1 max(X_scaled(:,1))+1 ...
         min(X_scaled(:,2))-1 max(X_scaled(:,2))+1], 30);
    contourf(gca, 'LevelList', [0.05], 'FaceAlpha', 0.1, ...
        'EdgeColor', colors(i,:), 'LineWidth', 2);
end

title('GMM高斯分布轮廓（95%置信区间）');
xlabel('特征1');
ylabel('特征2');
grid on;
hold off;

% 展示样本的概率分布（选择3个样本）
sample_indices = [10, 100, 200];
figure('Position', [100 200 800 400]);
bar_width = 0.25;
for i = 1:length(sample_indices)
    idx = sample_indices(i);
    bar((1:k) + (i-1)*bar_width, probs(idx, :), bar_width, ...
        'FaceColor', colors(i,:), 'DisplayName', sprintf('样本 %d', idx));
    hold on;
end

set(gca, 'XTick', (1:k) + bar_width, 'XTickLabel', arrayfun(@(x) sprintf('簇 %d', x), 0:k-1, 'UniformOutput', false));
ylabel('概率');
title('样本属于各簇的概率分布');
legend('Location', 'best');
grid on;
hold off;
    