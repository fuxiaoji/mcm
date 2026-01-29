% 高斯回归预测模型案例说明：
% 本案例使用高斯过程回归(GPR)预测某化学反应的产物浓度，
% 基于反应温度和反应时间两个特征。高斯过程是一种非参数模型，
% 能够捕捉数据中的非线性模式，并提供预测的不确定性估计。
% 本案例使用径向基函数(RBF)作为核函数，适用于具有平滑变化特性的数据。

% 1. 准备数据
% 设置随机种子，确保结果可复现
rng(42);

% 特征数据：[反应温度(℃), 反应时间(小时)]
X = [
    50, 1; 50, 2; 50, 3; 50, 4;
    60, 1; 60, 2; 60, 3; 60, 4;
    70, 1; 70, 2; 70, 3; 70, 4;
    80, 1; 80, 2; 80, 3; 80, 4
];

% 目标数据：产物浓度(%)，与特征呈非线性关系
y = [
    22.3; 35.6; 45.2; 50.1;
    30.5; 48.2; 60.3; 65.8;
    45.8; 62.5; 72.1; 76.5;
    55.2; 70.3; 80.1; 83.6
];

% 2. 数据预处理：标准化
[X_scaled, X_mu, X_sigma] = zscore(X);
[y_scaled, y_mu, y_sigma] = zscore(y);

% 3. 划分训练集和测试集
train_indices = [1, 2, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16];  % 训练集索引(1-based)
test_indices = [3, 6, 10, 13];                              % 测试集索引(1-based)
X_train = X_scaled(train_indices, :);
y_train = y_scaled(train_indices, :);
X_test = X_scaled(test_indices, :);
y_test = y_scaled(test_indices, :);

% 4. 定义并训练高斯过程回归模型
% 创建高斯过程回归模型，使用RBF核函数
gpr = fitrgp(X_train, y_train, ...
    'KernelFunction', 'RBF', ...       % 径向基函数核
    'KernelParameters', [1, 1],       % 初始核参数
    'Sigma', 0.01,                    % 噪声标准差
    'OptimizeHyperparameters', 'auto', % 自动优化超参数
    'HyperparameterOptimizationOptions', struct('MaxIterations', 50));

% 输出优化后的核参数
disp("优化后的核参数:");
disp(gpr.KernelParameters);

% 5. 模型预测
% 对训练集和测试集进行预测，同时获取标准差
[y_train_pred, y_train_std] = predict(gpr, X_train);
[y_test_pred, y_test_std] = predict(gpr, X_test);

% 反标准化预测结果
y_train_pred = y_pred_denorm(y_train_pred, y_mu, y_sigma);
y_test_pred = y_pred_denorm(y_test_pred, y_mu, y_sigma);
y_train_actual = y_pred_denorm(y_train, y_mu, y_sigma);
y_test_actual = y_pred_denorm(y_test, y_mu, y_sigma);

% 调整标准差到原始尺度
y_train_std_actual = y_train_std * y_sigma;
y_test_std_actual = y_test_std * y_sigma;

% 6. 模型评估
% 计算训练集和测试集的性能指标
train_rmse = sqrt(mean((y_train_actual - y_train_pred).^2));
test_rmse = sqrt(mean((y_test_actual - y_test_pred).^2));
train_r2 = 1 - sum((y_train_actual - y_train_pred).^2) / sum((y_train_actual - mean(y_train_actual)).^2);
test_r2 = 1 - sum((y_test_actual - y_test_pred).^2) / sum((y_test_actual - mean(y_test_actual)).^2);

% 7. 预测新数据点
% 新的反应条件
future_X = [55, 2.5; 65, 3.5; 75, 2; 85, 3];
% 标准化新数据
future_X_scaled = (future_X - X_mu) ./ X_sigma;
% 预测
[future_y_pred, future_y_std] = predict(gpr, future_X_scaled);
future_y_pred = y_pred_denorm(future_y_pred, y_mu, y_sigma);
future_y_std_actual = future_y_std * y_sigma;

% 8. 输出结果
disp("\n高斯过程回归预测模型结果：");
fprintf("训练集RMSE: %.4f, R²: %.4f\n", train_rmse, train_r2);
fprintf("测试集RMSE: %.4f, R²: %.4f\n", test_rmse, test_r2);

disp("\n训练集预测结果：");
for i = 1:size(y_train_actual, 1)
    fprintf("实际值: %.2f%%, 预测值: %.2f%%, 标准差: %.2f%%\n", ...
        y_train_actual(i), y_train_pred(i), y_train_std_actual(i));
end

disp("\n测试集预测结果：");
for i = 1:size(y_test_actual, 1)
    fprintf("实际值: %.2f%%, 预测值: %.2f%%, 标准差: %.2f%%\n", ...
        y_test_actual(i), y_test_pred(i), y_test_std_actual(i));
end

disp("\n新反应条件下的产物浓度预测：");
for i = 1:size(future_X, 1)
    fprintf("温度%.0f℃, 时间%.1f小时: 预测浓度%.2f%%, 标准差%.2f%%\n", ...
        future_X(i,1), future_X(i,2), future_y_pred(i), future_y_std_actual(i));
end

% 9. 可视化结果（选择温度为特征，固定时间为2小时）
figure;
% 创建温度序列
temp_range = linspace(45, 85, 100)';
time_fixed = 2;  % 固定时间为2小时
X_plot = [temp_range, ones(length(temp_range), 1) * time_fixed];
% 标准化
X_plot_scaled = (X_plot - X_mu) ./ X_sigma;

% 预测
[y_plot_pred, y_plot_std] = predict(gpr, X_plot_scaled);
y_plot_pred = y_pred_denorm(y_plot_pred, y_mu, y_sigma);
y_plot_std_actual = y_plot_std * y_sigma;

% 绘制结果
plot(temp_range, y_plot_pred, 'b-', 'LineWidth', 1.5, 'DisplayName', '预测均值');
hold on;
% 95%置信区间
fill([temp_range; flipud(temp_range)], ...
     [y_plot_pred - 1.96 * y_plot_std_actual; flipud(y_plot_pred + 1.96 * y_plot_std_actual)], ...
     'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '95%置信区间');

% 绘制实际数据点（时间=2小时）
mask = X(:, 2) == 2;
scatter(X(mask, 1), y(mask), 50, 'red', 'filled', 'DisplayName', '实际数据点');

xlabel('反应温度 (℃)');
ylabel('产物浓度 (%)');
title('固定反应时间为2小时的产物浓度预测');
legend();
grid on;
hold off;

% 反标准化函数
function y_denorm = y_pred_denorm(y_norm, y_mu, y_sigma)
    y_denorm = y_norm * y_sigma + y_mu;
end
