% T-SNE降维算法案例说明：
% 本案例使用t-SNE(t分布随机邻域嵌入)算法对高维数据进行降维。
% t-SNE是一种非线性降维方法，特别适合高维数据的可视化，能够较好地保留数据的局部结构。
% 示例中使用MNIST手写数字数据集(784维)，通过t-SNE降维到2维空间进行可视化。

% 检查并安装必要的工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox来运行t-SNE');
end

% 加载MNIST数据集（使用MATLAB内置数据集或自行下载）
% 注意：如果没有内置MNIST，可以从MathWorks文件交换下载或使用其他数据集
try
    % 尝试加载MATLAB示例数据
    load('digits.mat');  % 小型手写数字数据集(64维)
    X = digits.images;
    X = reshape(X, size(X,1)*size(X,2), size(X,3))';  % 转换为特征矩阵
    y = digits.labels + 1;  % 标签从1开始
    n_samples = min(500, size(X,1));  % 使用部分样本
    X_sample = X(1:n_samples, :);
    y_sample = y(1:n_samples);
    original_dim = size(X_sample, 2);
    dataset_name = '手写数字数据集';
catch
    error('无法加载数据集，请确保已安装必要的数据或修改代码使用其他数据集');
end

% 数据标准化
X_scaled = zscore(X_sample);  % 标准化：均值为0，标准差为1

% 应用t-SNE降维到2维
fprintf('运行t-SNE降维...\n');
params.Perplexity = 30;       % 困惑度，通常在5-50之间
params.LearnRate = 200;       % 学习率
params.MaxIterations = 1000;  % 迭代次数
params.Seed = 42;             % 随机种子，确保结果可复现
[X_tsne, ~] = tsne(X_scaled, params);

% 输出t-SNE结果信息
fprintf('\nt-SNE降维结果：\n');
fprintf('原始数据维度: %d 维\n', original_dim);
fprintf('降维后数据维度: %d 维\n', size(X_tsne, 2));
fprintf('使用参数: perplexity=%d, learning_rate=%d, n_iter=%d\n', ...
        params.Perplexity, params.LearnRate, params.MaxIterations);

% 可视化t-SNE降维结果
figure('Position', [100 100 800 700]);
colors = lines(10);  % 10种颜色
markers = {'o', 's', '^', 'D', '*', 'P', 'X', 'd', 'p', 'h'};  % 10种标记

for digit = 1:10
    idx = y_sample == digit;
    plot(X_tsne(idx, 1), X_tsne(idx, 2), 'Marker', markers{digit}, ...
         'MarkerFaceColor', colors(digit,:), 'MarkerEdgeColor', 'k', ...
         'MarkerSize', 8, 'LineStyle', 'none', 'DisplayName', num2str(digit-1));
    hold on;
end

legend('Location', 'eastoutside', 'Title', '数字');
title(sprintf('t-SNE降维：%s(%d维→2维)', dataset_name, original_dim));
xlabel('t-SNE特征1');
ylabel('t-SNE特征2');
grid on;
box on;
hold off;
    