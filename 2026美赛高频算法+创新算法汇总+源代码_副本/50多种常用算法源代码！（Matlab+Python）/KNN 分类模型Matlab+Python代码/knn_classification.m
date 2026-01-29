% KNN分类模型案例说明：
% KNN（K近邻）是一种简单的监督学习算法，其核心思想是：一个样本的类别由其周围k个最近邻样本的类别决定。
% KNN没有显式的训练过程，属于"懒惰学习"算法，在预测时才进行计算。
%
% 算法步骤：
% 1. 确定k值（近邻数量）
% 2. 计算待预测样本与所有训练样本的距离（常用欧氏距离）
% 3. 选取距离最近的k个样本
% 4. 这k个样本中出现次数最多的类别即为待预测样本的类别
%
% 本案例使用鸢尾花数据集，通过KNN模型进行分类，并对比不同k值对模型性能的影响，
% 最终选择最优k值并可视化分类结果。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行KNN分类');
end

% 加载数据集
load fisheriris;
X = meas;  % 特征数据
y = species;  % 标签
class_names = unique(y);
feature_names = {'花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'};

% 数据划分：训练集和测试集（7:3）
rng(42);  % 设置随机种子
cv = cvpartition(y, 'HoldOut', 0.3);  % 保留30%作为测试集
X_train = X(cv.training, :);
y_train = y(cv.training, :);
X_test = X(cv.test, :);
y_test = y(cv.test, :);

% 数据标准化（KNN对距离敏感，需要标准化）
[X_train_scaled, mu, sigma] = zscore(X_train);  % 训练集标准化
X_test_scaled = (X_test - mu) ./ sigma;  % 测试集标准化（使用训练集的均值和标准差）

% 尝试不同的k值，选择最优k
k_range = 1:20;
accuracy_scores = zeros(size(k_range));

for i = 1:length(k_range)
    k = k_range(i);
    % 创建并训练KNN模型
    knn_model = fitcknn(X_train_scaled, y_train, ...
        'NumNeighbors', k, ...
        'Distance', 'euclidean');
    
    % 预测并计算准确率
    y_pred = predict(knn_model, X_test_scaled);
    accuracy = mean(strcmp(y_pred, y_test));
    accuracy_scores(i) = accuracy;
    fprintf("k=%d时，测试集准确率: %.4f\n", k, accuracy);
end

% 找到最优k值
[~, best_idx] = max(accuracy_scores);
best_k = k_range(best_idx);
fprintf("\n最优k值为: %d\n", best_k);

% 使用最优k值构建最终模型
knn_best = fitcknn(X_train_scaled, y_train, ...
    'NumNeighbors', best_k, ...
    'Distance', 'euclidean');
y_pred_best = predict(knn_best, X_test_scaled);

% 输出详细评估结果
fprintf("\n最优模型在测试集上的性能：\n");
fprintf("准确率: %.4f\n", mean(strcmp(y_pred_best, y_test)));
fprintf("\n混淆矩阵：\n");
conf_mat = confusionmat(y_test, y_pred_best);
disp(conf_mat);
fprintf("\n分类报告：\n");
disp(classification_report(y_test, y_pred_best));

% 可视化不同k值的准确率
figure('Position', [100 100 800 500]);
plot(k_range, accuracy_scores, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('k值（近邻数量）');
ylabel('测试集准确率');
title('不同k值对KNN模型性能的影响');
grid on;
xline(best_k, 'r--', sprintf('最优k=%d', best_k), 'LineWidth', 1.5);
legend('准确率', '最优k值');

% 使用PCA降维到2D，可视化分类结果
[coeff, score, latent] = pca(X_train_scaled);
X_train_pca = score(:, 1:2);  % 取前两个主成分
X_test_pca = (X_test_scaled - mu) * coeff(:, 1:2);  % 转换测试集到PCA空间

% 绘制决策边界
h = 0.02;  % 网格步长
x_min = min(X_train_pca(:, 1)) - 1;
x_max = max(X_train_pca(:, 1)) + 1;
y_min = min(X_train_pca(:, 2)) - 1;
y_max = max(X_train_pca(:, 2)) + 1;
[xx, yy] = meshgrid(x_min:h:x_max, y_min:h:y_max);

% 使用PCA转换后的训练数据重新训练模型（仅用于可视化）
knn_pca = fitcknn(X_train_pca, y_train, 'NumNeighbors', best_k);
Z = predict(knn_pca, [xx(:) yy(:)]);
Z = reshape(cell2mat(Z), size(xx));

% 绘制决策边界和样本点
figure('Position', [100 100 1000 500]);

% 1. 训练集分类结果
subplot(1, 2, 1);
gscatter(xx(:), yy(:), Z, [], '.', 1);  % 决策区域
hold on;
gscatter(X_train_pca(:, 1), X_train_pca(:, 2), y_train, ...
    'rgb', 'osd', 8, 'off');  % 训练样本
title(sprintf('KNN决策边界 (k=%d) - 训练集', best_k));
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
legend(class_names, 'Location', 'best');
grid on;
hold off;

% 2. 测试集分类结果
subplot(1, 2, 2);
gscatter(xx(:), yy(:), Z, [], '.', 1);  % 决策区域
hold on;
% 绘制真实标签（方形）和预测标签（圆形）
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_test, ...
    'rgb', 'sss', 10, 'off');  % 真实标签
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_pred_best, ...
    'rgb', 'ooo', 6, 'off');  % 预测标签
title(sprintf('KNN决策边界 (k=%d) - 测试集', best_k));
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
legend([class_names; {'真实标签'; '预测标签'}], 'Location', 'best');
grid on;
hold off;
    