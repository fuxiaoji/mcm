% 决策树分类模型案例说明：
% 决策树是一种基于树状结构进行决策的监督学习算法，通过一系列if-then规则对数据进行分类。
% 每个内部节点表示一个特征的判断，每个分支代表一个判断结果，每个叶节点代表一个类别。
%
% 算法步骤：
% 1. 选择最佳特征作为根节点，根据该特征的不同取值创建分支
% 2. 对每个分支递归地应用步骤1，选择最佳特征继续分裂
% 3. 当满足停止条件（如节点样本数小于阈值、树深度达到上限等）时停止分裂
% 4. 叶节点的类别为该节点中样本数最多的类别
%
% 本案例使用红酒数据集，构建决策树模型预测红酒质量等级，并可视化决策树结构，
% 分析特征重要性，对比不同树深度对模型性能的影响。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行决策树分类');
end

% 加载数据集
load wine;
X = meas;  % 特征数据
y = species;  % 标签（3个类别）
class_names = unique(y);
feature_names = {'酒精含量', '苹果酸', '灰分', '灰分碱度', '镁含量', ...
                 '总酚', '类黄酮', '非黄烷类酚', '原花青素', '颜色强度', ...
                 '色调', '稀释葡萄酒的OD280/OD315', '脯氨酸'};

% 数据划分：训练集和测试集（7:3）
rng(42);  % 设置随机种子
cv = cvpartition(y, 'HoldOut', 0.3);  % 保留30%作为测试集
X_train = X(cv.training, :);
y_train = y(cv.training, :);
X_test = X(cv.test, :);
y_test = y(cv.test, :);

% 尝试不同的树深度，选择最优深度
max_depth_range = 1:10;
accuracy_scores = zeros(size(max_depth_range));

for i = 1:length(max_depth_range)
    depth = max_depth_range(i);
    % 创建并训练决策树模型
    dt_model = fitctree(X_train, y_train, ...
        'MaxDepth', depth, ...
        'SplitCriterion', 'gini', ...  % 使用基尼不纯度
        'RandomSeed', 42);
    
    % 预测并计算准确率
    y_pred = predict(dt_model, X_test);
    accuracy = mean(strcmp(y_pred, y_test));
    accuracy_scores(i) = accuracy;
    fprintf("树深度=%d时，测试集准确率: %.4f\n", depth, accuracy);
end

% 找到最优树深度
[~, best_idx] = max(accuracy_scores);
best_depth = max_depth_range(best_idx);
fprintf("\n最优树深度为: %d\n", best_depth);

% 使用最优树深度构建最终模型
dt_best = fitctree(X_train, y_train, ...
    'MaxDepth', best_depth, ...
    'SplitCriterion', 'gini', ...
    'RandomSeed', 42);
y_pred_best = predict(dt_best, X_test);

% 输出评估结果
fprintf("\n最优决策树模型在测试集上的性能：\n");
accuracy = mean(strcmp(y_pred_best, y_test));
fprintf("准确率: %.4f\n", accuracy);

fprintf("\n混淆矩阵：\n");
conf_mat = confusionmat(y_test, y_pred_best);
disp(conf_mat);

fprintf("\n分类报告：\n");
disp(classification_report(y_test, y_pred_best));

% 输出特征重要性
feature_importance = dt_best.FeatureImportance;
[~, idx] = sort(feature_importance, 'descend');  % 按重要性降序排列

fprintf("\n特征重要性排序：\n");
for i = 1:length(idx)
    fprintf("%s: %.4f\n", feature_names{idx(i)}, feature_importance(idx(i)));
end

% 可视化不同树深度的准确率
figure('Position', [100 100 800 500]);
plot(max_depth_range, accuracy_scores, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('树深度');
ylabel('测试集准确率');
title('不同树深度对决策树模型性能的影响');
grid on;
xline(best_depth, 'r--', sprintf('最优深度=%d', best_depth), 'LineWidth', 1.5);
legend('准确率', '最优深度');

% 可视化决策树结构
figure('Position', [100 100 1200 800]);
view(dt_best, 'Mode', 'graph');
title(sprintf('决策树结构 (深度=%d)', best_depth));

% 可视化特征重要性
figure('Position', [100 200 800 600]);
barh(feature_importance(idx), 'FaceColor', 'skyblue');
set(gca, 'YTick', 1:length(idx), ...
         'YTickLabel', feature_names(idx), ...
         'FontName', 'SimHei');  % 支持中文显示
xlabel('重要性');
ylabel('特征');
title('决策树特征重要性');
grid on;
box on;

% 使用PCA降维到2D，可视化分类结果
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

% 使用PCA转换后的训练数据重新训练模型（仅用于可视化）
dt_pca = fitctree(X_train_pca, y_train, ...
    'MaxDepth', best_depth, ...
    'SplitCriterion', 'gini', ...
    'RandomSeed', 42);
Z = predict(dt_pca, [xx(:) yy(:)]);
Z = reshape(cell2mat(Z), size(xx));

% 绘制决策边界和样本点
figure('Position', [100 100 800 600]);
gscatter(xx(:), yy(:), Z, [], '.', 1);  % 决策区域
hold on;
% 绘制真实标签（方形）和预测标签（圆形）
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_test, ...
    'rgb', 'sss', 10, 'off');  % 真实标签
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_pred_best, ...
    'rgb', 'ooo', 6, 'off');  % 预测标签
title(sprintf('决策树决策边界 (深度=%d) - 测试集', best_depth));
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
legend([class_names; {'真实标签'; '预测标签'}], 'Location', 'best');
grid on;
hold off;
    