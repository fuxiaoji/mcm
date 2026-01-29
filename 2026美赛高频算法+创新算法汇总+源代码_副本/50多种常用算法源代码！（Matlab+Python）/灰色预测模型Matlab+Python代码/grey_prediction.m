% 灰色预测模型案例说明：
% 本案例使用GM(1,1)灰色预测模型预测某企业未来3年的销售额，
% 已知过去6年的销售额数据（单位：万元）。灰色预测模型适用于
% 数据量少（通常4-10个数据）、信息不完全的预测问题，
% 不需要数据满足典型分布特征。

% 1. 准备数据
% 原始数据序列（过去6年的销售额，单位：万元）
x0 = [120, 135, 150, 170, 190, 215];
n = length(x0);  % 数据长度

% 2. 累加生成（1-AGO）
x1 = cumsum(x0);  % 一次累加序列

% 3. 构造数据矩阵B和数据向量Y
B = zeros(n-1, 2);
Y = zeros(n-1, 1);

for i = 1:n-1
    B(i, 1) = -0.5 * (x1(i) + x1(i+1));  % 均值生成
    B(i, 2) = 1;
    Y(i, 1) = x0(i+1);
end

% 4. 计算模型参数a和b（最小二乘法）
% 参数估计：(a, b)^T = (B^T B)^(-1) B^T Y
BT = B';
params = (BT * B) \ (BT * Y);
a = params(1);  % 发展系数
b = params(2);  % 灰作用量

% 5. 计算历史数据拟合值
x0_hat = zeros(1, n);
x0_hat(1) = x0(1);  % 第一个值等于原始值
for i = 2:n
    % 时间响应函数：x1_hat(k) = (x0(1) - b/a) * exp(-a*(k-1)) + b/a
    x1_hat_k = (x0(1) - b/a) * exp(-a*(i-1)) + b/a;
    x1_hat_k_prev = (x0(1) - b/a) * exp(-a*(i-2)) + b/a;
    x0_hat(i) = x1_hat_k - x1_hat_k_prev;  % 累减得到原始序列预测值
end

% 6. 预测未来3年数据
forecast_num = 3;  % 预测未来3年
future_x0 = zeros(1, forecast_num);
for i = 1:forecast_num
    k = n + i;  % 预测第k年
    x1_hat_k = (x0(1) - b/a) * exp(-a*(k-1)) + b/a;
    x1_hat_k_prev = (x0(1) - b/a) * exp(-a*(k-2)) + b/a;
    future_x0(i) = x1_hat_k - x1_hat_k_prev;
end

% 7. 模型检验：计算后验差比C和小误差概率P
% 计算残差
epsilon = x0 - x0_hat;
% 原始数据标准差
s1 = std(x0, 1);
% 残差标准差
s2 = std(epsilon, 1);
% 后验差比
C = s2 / s1;
% 小误差概率
P = mean(abs(epsilon - mean(epsilon)) < 0.6745 * s1);

% 8. 输出结果
disp("GM(1,1)灰色预测模型结果：");
fprintf("模型参数：a=%.6f, b=%.6f\n", a, b);
fprintf("后验差比C=%.4f, 小误差概率P=%.4f\n", C, P);
if C < 0.35 && P > 0.95
    disp("模型精度等级：好");
elseif C < 0.5 && P > 0.8
    disp("模型精度等级：合格");
elseif C < 0.65 && P > 0.7
    disp("模型精度等级：勉强");
else
    disp("模型精度等级：不合格");
end

disp("\n历史数据拟合结果：");
for i = 1:n
    fprintf("第%d年: 实际值=%.0f, 拟合值=%.2f, 残差=%.2f\n", ...
        i, x0(i), x0_hat(i), epsilon(i));
end

disp("\n未来3年预测结果：");
for i = 1:forecast_num
    fprintf("第%d年预测值: %.2f\n", n+i, future_x0(i));
end

% 9. 可视化结果
figure;
plot(1:n, x0, 'bo-', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', '实际销售额');
hold on;
plot(1:n, x0_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', '拟合销售额');
plot(n+1:n+forecast_num, future_x0, 'g*-', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', '预测销售额');
xlabel('年份');
ylabel('销售额（万元）');
title('GM(1,1)灰色预测模型');
legend();
grid on;
hold off;
