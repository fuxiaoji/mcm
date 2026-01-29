% 朴素贝叶斯分类模型案例说明：
% 朴素贝叶斯是基于贝叶斯定理和特征条件独立假设的分类算法。"朴素"指的是假设各个特征之间相互独立，
% 这一假设简化了计算，使模型能够高效处理高维数据。
%
% 算法步骤：
% 1. 计算先验概率：每个类别的出现概率
% 2. 计算似然概率：给定类别时每个特征的条件概率
% 3. 应用贝叶斯定理计算后验概率：给定特征时样本属于某个类别的概率
% 4. 选择后验概率最大的类别作为预测结果
%
% 本案例使用鸢尾花数据集（连续特征）和手写数字数据集（离散特征），
% 分别构建高斯朴素贝叶斯和多项式朴素贝叶斯模型，并评估其分类性能。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行朴素贝叶斯分类');
end

%% 1. 高斯朴素贝叶斯（处理连续特征）- 使用鸢尾花数据集
fprintf('=== 高斯朴素贝叶斯（鸢尾花数据集） ===\n');

% 加载数据集
load fisheriris;
X = meas;  % 特征数据（连续值）
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

% 训练高斯朴素贝叶斯模型
gnb = fitcnb(X_train, y_train, 'DistributionNames', {'normal'});

% 预测
y_pred_gnb = predict(gnb, X_test);

% 输出评估结果
accuracy_gnb = mean(strcmp(y_pred_gnb, y_test));
fprintf('高斯朴素贝叶斯测试集准确率: %.4f\n', accuracy_gnb);

fprintf('\n混淆矩阵：\n');
conf_mat_gnb = confusionmat(y_test, y_pred_gnb);
disp(conf_mat_gnb);

fprintf('\n分类报告：\n');
disp(classification_report(y_test, y_pred_gnb));

%% 2. 多项式朴素贝叶斯（处理离散特征）- 使用手写数字数据集
fprintf('\n=== 多项式朴素贝叶斯（手写数字数据集） ===\n');

% 加载数据集
load digits;
X = double(digits);  % 特征数据（0-255的像素值）
y = digitlabels;     % 标签（0-9）
class_names_digits = arrayfun(@(x) sprintf('数字%d', x), 0:9, 'UniformOutput', false);

% 数据预处理：将像素值二值化（0或1）以适应多项式朴素贝叶斯
X_binary = X > 127;  % 大于127的像素视为1，否则为0

% 数据划分：训练集和测试集（7:3）
cv_digits = cvpartition(y, 'HoldOut', 0.3);  % 保留30%作为测试集
X_train_digits = X_binary(cv_digits.training, :);
y_train_digits = y(cv_digits.training, :);
X_test_digits = X_binary(cv_digits.test, :);
y_test_digits = y(cv_digits.test, :);

% 训练多项式朴素贝叶斯模型
mnb = fitcnb(X_train_digits, y_train_digits, ...
    'DistributionNames', {'multinomial'}, ...
    'Prior', 'empirical');

% 预测
y_pred_mnb = predict(mnb, X_test_digits);

% 输出评估结果
accuracy_mnb = mean(y_pred_mnb == y_test_digits);
fprintf('多项式朴素贝叶斯测试集准确率: %.4f\n', accuracy_mnb);

fprintf('\n混淆矩阵（前5类）：\n');
conf_mat_mnb = confusionmat(y_test_digits, y_pred_mnb);
disp(conf_mat_mnb(1:5, 1:5));  % 只显示前5类以简化输出

%% 可视化结果

% 1. 高斯朴素贝叶斯决策边界（使用PCA降维）
[coeff, score, latent] = pca(X_train);
X_train_pca = score(:, 1:2);  % 取前两个主成分
X_test_pca = (X_test - mean(X_train)) * coeff(:, 1:2);  % 转换测试集到PCA空间

% 绘制决策边界
h = 0.05;  % 网格步长
x_min = min(X_train_pca(:, 1)) - 1;
x_max = max(X_train_pca(:, 1)) + 1;
y_min = min(X_train_pca(:, 2)) - 1;
y_max = max(X_train_pca(:, 2)) + 1;
[xx, yy] = meshgrid(x_min:h:x_max, y_min:h:y_max);

% 将PCA空间的点映射回原始特征空间（近似）
grid_points = [xx(:) yy(:)] * coeff(:, 1:2)' + mean(X_train);

% 预测网格点类别
Z = predict(gnb, grid_points);
Z = reshape(cell2mat(Z), size(xx));

% 绘制决策边界和样本点
figure('Position', [100 100 800 600]);
gscatter(xx(:), yy(:), Z, [], '.', 1);  % 决策区域
hold on;
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_test, ...
    'rgb', 'osd', 8, 'off');  % 测试样本
title('高斯朴素贝叶斯决策边界 - 鸢尾花数据集');
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
legend(class_names, 'Location', 'best');
grid on;
hold off;

% 2. 多项式朴素贝叶斯在手写数字上的预测结果
figure('Position', [100 200 800 600]);
num_examples = 16;
indices = randperm(length(y_test_digits), num_examples);
for i = 1:num_examples
    subplot(4, 4, i);
    img = reshape(X_test(indices(i), :), [8 8])';  % 还原为8x8图像
    imshow(img, []);
    title(sprintf('真实: %d, 预测: %d', ...
        y_test_digits(indices(i)), y_pred_mnb(indices(i))));
    axis off;
end
sgtitle('多项式朴素贝叶斯对手写数字的预测结果');

% 3. 两种朴素贝叶斯模型的准确率对比
figure('Position', [100 100 600 400]);
models = {'高斯朴素贝叶斯', '多项式朴素贝叶斯'};
accuracies = [accuracy_gnb, accuracy_mnb];
bar(accuracies, 'FaceColor', 'flat', 'CData', [0.6 0.8 1; 0.6 1 0.6]);
set(gca, 'XTickLabel', models, 'FontName', 'SimHei');  % 支持中文显示
ylim([0, 1.0]);
ylabel('测试集准确率');
title('不同朴素贝叶斯模型的性能对比');
for i = 1:length(accuracies)
    text(i, accuracies(i) + 0.01, sprintf('%.4f', accuracies(i)), ...
        'HorizontalAlignment', 'center');
end
grid on;
    