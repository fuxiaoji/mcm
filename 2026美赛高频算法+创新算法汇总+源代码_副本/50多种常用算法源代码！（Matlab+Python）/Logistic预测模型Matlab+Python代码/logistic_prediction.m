% Logistic预测模型案例说明：
% 本案例使用Logistic模型预测某地区未来5年的人口数量，
% 已知过去10年的人口数据（单位：万人），Logistic模型适用于
% 具有饱和增长特性的预测问题，如人口增长、产品扩散等。
% 模型公式：y(t) = K / (1 + (K/y0 - 1) * exp(-r*t))
% 其中K为环境承载力，r为增长率，y0为初始值。

% 1. 准备数据
% 年份（相对值，0表示起始年份）
t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
% 对应年份的人口数量（单位：万人）
y = [32.5, 35.6, 39.0, 42.8, 46.9, 51.3, 55.9, 60.5, 65.0, 69.2];

% 2. 定义Logistic模型（匿名函数）
logistic_func = @(params, t) params(1) ./ (1 + (params(1)/params(3) - 1) * exp(-params(2) * t));

% 3. 定义误差函数（用于最小二乘拟合）
error_func = @(params, t, y) logistic_func(params, t) - y;

% 4. 拟合模型参数
% 初始参数估计（K: 环境承载力，r: 增长率，y0: 初始值）
initial_guess = [100, 0.1, 30];
% 最小二乘拟合
opts = optimset('MaxFunctionEvaluations', 10000);
params = lsqnonlin(error_func, initial_guess, [], [], opts, t, y);
K = params(1);  % 环境承载力
r = params(2);  % 增长率
y0 = params(3); % 初始值

% 5. 计算拟合值和预测值
y_pred = logistic_func(params, t);  % 历史数据拟合值
% 预测未来5年数据
future_t = 10:14;  % 未来5年的时间点
future_y = logistic_func(params, future_t);  % 预测值

% 6. 计算拟合优度R²
ss_total = sum((y - mean(y)).^2);
ss_residual = sum((y - y_pred).^2);
r_squared = 1 - (ss_residual / ss_total);

% 7. 输出结果
disp("Logistic预测模型结果：");
fprintf("模型参数：K=%.2f, r=%.4f, y0=%.2f\n", K, r, y0);
fprintf("拟合优度R²：%.4f\n", r_squared);
disp("\n历史数据拟合值：");
for i = 1:length(t)
    fprintf("年份%d: 实际值=%.1f, 拟合值=%.2f\n", t(i), y(i), y_pred(i));
end
disp("\n未来5年预测值：");
for i = 1:length(future_t)
    fprintf("年份%d: 预测值=%.2f\n", future_t(i), future_y(i));
end

% 8. 可视化结果
figure;
plot(t, y, 'bo', 'MarkerSize', 8, 'DisplayName', '实际数据');
hold on;
plot(t, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', '拟合曲线');
plot(future_t, future_y, 'g--', 'LineWidth', 2, 'DisplayName', '预测曲线');
xlabel('年份（相对值）');
ylabel('人口数量（万人）');
title('Logistic模型人口预测');
legend();
grid on;
hold off;
