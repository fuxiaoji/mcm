% 二次指数平滑预测模型案例说明：
% 本案例使用二次指数平滑法预测某产品未来4个月的销售量，
% 已知过去12个月的销售数据（单位：件）。二次指数平滑适用于
% 具有线性趋势的数据预测，在一次指数平滑基础上增加了对趋势的平滑处理。
% 平滑系数α的取值范围为0-1，通常通过试算选择使预测误差最小的值。

% 1. 准备数据
% 过去12个月的销售量（单位：件）
y = [120, 135, 142, 150, 165, 178, 190, 205, 218, 230, 245, 260];
n = length(y);  % 数据长度

% 2. 设置平滑系数
alpha = 0.3;  % 平滑系数，可根据实际情况调整

% 3. 初始化一次和二次指数平滑值
s1 = zeros(1, n);  % 一次指数平滑值
s2 = zeros(1, n);  % 二次指数平滑值

s1(1) = y(1);  % 第一个一次平滑值等于原始数据
s2(1) = y(1);  % 第一个二次平滑值等于原始数据

% 4. 计算一次和二次指数平滑值
for i = 2:n
    % 一次指数平滑公式：s1(i) = α*y(i) + (1-α)*s1(i-1)
    s1(i) = alpha * y(i) + (1 - alpha) * s1(i-1);
    % 二次指数平滑公式：s2(i) = α*s1(i) + (1-α)*s2(i-1)
    s2(i) = alpha * s1(i) + (1 - alpha) * s2(i-1);
end

% 5. 计算平滑系数和预测模型
% 估计当前水平和趋势
a = 2 * s1(n) - s2(n);  % 截距项
b = (alpha / (1 - alpha)) * (s1(n) - s2(n));  % 趋势项

% 6. 计算历史数据拟合值
y_hat = zeros(1, n);
for t = 1:n
    % 第t期的拟合值
    y_hat(t) = a - b * (n - t);
end

% 7. 预测未来4个月数据
forecast_num = 4;  % 预测未来4个月
future_t = 1:forecast_num;  % 预测步数
future_y = a + b * future_t;  % 预测公式：y_hat(n+k) = a + b*k

% 8. 计算预测误差（均方根误差RMSE）
rmse = sqrt(mean((y - y_hat).^2));

% 9. 输出结果
disp("二次指数平滑预测模型结果：");
fprintf("平滑系数α = %.1f\n", alpha);
fprintf("模型参数：a = %.2f, b = %.2f\n", a, b);
fprintf("均方根误差RMSE = %.2f\n", rmse);

disp("\n历史数据拟合结果：");
for i = 1:n
    fprintf("第%d月: 实际值=%d, 拟合值=%.2f, 误差=%.2f\n", ...
        i, y(i), y_hat(i), y(i)-y_hat(i));
end

disp("\n未来4个月预测结果：");
for i = 1:forecast_num
    fprintf("第%d月预测值: %.2f\n", n+i, future_y(i));
end

% 10. 可视化结果
figure;
plot(1:n, y, 'bo-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '实际销售量');
hold on;
plot(1:n, y_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', '拟合销售量');
plot(n+1:n+forecast_num, future_y, 'g*-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '预测销售量');
xlabel('月份');
ylabel('销售量（件）');
title(sprintf('二次指数平滑预测 (α=%.1f)', alpha));
legend();
grid on;
hold off;
