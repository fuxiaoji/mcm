% ARMA时间序列预测模型案例说明：
% ARMA（自回归移动平均模型）是一种常用的时间序列分析方法，结合了AR（自回归）和MA（移动平均）模型的特点。
% ARMA(p,q)模型表示一个时间序列可以由其自身的p阶滞后项和q阶移动平均误差项来解释。

% 模型公式：
% y_t = c + φ_1 y_{t-1} + ... + φ_p y_{t-p} + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
% 其中：
% - y_t 是t时刻的序列值
% - c 是常数项
% - φ_i 是自回归系数
% - θ_j 是移动平均系数
% - ε_t 是白噪声序列

% 本案例使用一个模拟的包含趋势和季节性的时间序列数据，展示ARMA模型的构建和预测过程：
% 1. 生成模拟数据
% 2. 确定ARMA模型的最佳阶数(p,q)
% 3. 拟合ARMA模型
% 4. 进行预测并评估预测效果

% 清 workspace 和 command window
clear;
clc;

%% 1. 生成模拟时间序列数据
% 设置随机数种子，保证结果可重复
rng(123);

% 生成时间点（100个数据点）
t = 1:100;

% 生成包含趋势、季节性和噪声的时间序列
trend = 0.5 * t;  % 线性趋势
seasonality = 5 * sin(2 * pi * t / 12);  % 季节性成分（周期12）
noise = randn(size(t)) * 1.5;  % 高斯白噪声
y = trend + seasonality + noise;  % 总序列

% 可视化原始时间序列
figure('Position', [100 100 800 400]);
plot(t, y, 'b-', 'LineWidth', 1.5);
xlabel('时间', 'FontSize', 12);
ylabel('序列值', 'FontSize', 12);
title('原始时间序列数据', 'FontSize', 14);
grid on;
hold off;

%% 2. 序列平稳性检验与处理
% 对序列进行平稳性检验（ADF检验）
[h, pValue] = adftest(y);
fprintf('ADF平稳性检验结果：\n');
fprintf('原假设：序列存在单位根（非平稳）\n');
fprintf('检验结果：%s (p值 = %.4f)\n', ...
    h?'拒绝原假设（序列平稳）':'不能拒绝原假设（序列可能非平稳）', pValue);

% 如果序列非平稳，进行差分处理
if ~h
    y_diff = diff(y);  % 一阶差分
    [h_diff, pValue_diff] = adftest(y_diff);
    fprintf('一阶差分后ADF检验结果：%s (p值 = %.4f)\n', ...
        h_diff?'拒绝原假设（序列平稳）':'不能拒绝原假设（序列可能非平稳）', pValue_diff);
    y_stationary = y_diff;
else
    y_stationary = y;
end

%% 3. 确定ARMA模型的最佳阶数
% 计算自相关函数(ACF)和偏自相关函数(PACF)
figure('Position', [100 100 800 400]);
subplot(2,1,1);
autocorr(y_stationary, 20);
title('自相关函数(ACF)', 'FontSize', 12);
subplot(2,1,2);
parcorr(y_stationary, 20);
title('偏自相关函数(PACF)', 'FontSize', 12);
sgtitle('ACF和PACF图（用于确定ARMA阶数）', 'FontSize', 14);

% 尝试不同阶数的ARMA模型，选择AIC最小的模型
max_p = 3;  % 最大AR阶数
max_q = 3;  % 最大MA阶数
aic_values = zeros(max_p+1, max_q+1);  % 存储AIC值

for p = 0:max_p
    for q = 0:max_q
        try
            % 拟合ARMA(p,q)模型
            model = arima(p, 0, q);
            [est_model, ~, logL] = estimate(model, y_stationary);
            % 计算AIC值
            aic = -2*logL + 2*(p+q+1);  % +1是因为包含常数项
            aic_values(p+1, q+1) = aic;
        catch
            aic_values(p+1, q+1) = Inf;
        end
    end
end

% 找到AIC最小的模型阶数
[min_aic, idx] = min(aic_values(:));
[p_opt, q_opt] = ind2sub(size(aic_values), idx);
p_opt = p_opt - 1;  % 转换回0-based索引
q_opt = q_opt - 1;

fprintf('\nARMA模型阶数选择结果：\n');
fprintf('最佳ARMA阶数: ARMA(%d, %d)\n', p_opt, q_opt);
fprintf('最小AIC值: %.4f\n', min_aic);

%% 4. 拟合ARMA模型并进行预测
% 划分训练集和测试集（前80%用于训练，后20%用于测试）
train_ratio = 0.8;
n_train = floor(length(y) * train_ratio);
y_train = y(1:n_train);
y_test = y(n_train+1:end);

% 拟合最佳ARMA模型
model = arima(p_opt, 0, q_opt);  % 0表示不需要差分
est_model = estimate(model, y_train);

% 预测未来数据（与测试集长度相同）
n_forecast = length(y_test);
[forecast, ~, forecast_int] = forecast(est_model, n_forecast, 'Y0', y_train);

% 如果之前进行了差分，需要还原差分
if ~h
    % 还原一阶差分
    forecast = cumsum([y_train(end); forecast])';
    forecast = forecast(2:end);
end

%% 5. 评估预测效果并可视化
% 计算预测误差指标
mse = mean((forecast - y_test).^2);  % 均方误差
rmse = sqrt(mse);  % 均方根误差
mae = mean(abs(forecast - y_test));  % 平均绝对误差

fprintf('\n预测效果评估：\n');
fprintf('均方误差(MSE): %.4f\n', mse);
fprintf('均方根误差(RMSE): %.4f\n', rmse);
fprintf('平均绝对误差(MAE): %.4f\n', mae);

% 可视化预测结果
figure('Position', [100 100 800 400]);
plot(1:length(y_train), y_train, 'b-', 'LineWidth', 1.5);
hold on;
plot(length(y_train)+1:length(y), y_test, 'b-', 'LineWidth', 1.5, 'Alpha', 0.5);
plot(length(y_train)+1:length(y), forecast, 'r--', 'LineWidth', 2);
% 绘制预测区间
fill([length(y_train)+1:length(y), length(y):-1:length(y_train)+1], ...
     [forecast_int(:,1); forecast_int(end:-1:1,2)], ...
     'r', 'Alpha', 0.2, 'EdgeColor', 'none');
xlabel('时间', 'FontSize', 12);
ylabel('序列值', 'FontSize', 12);
title(sprintf('ARMA(%d,%d)模型预测结果', p_opt, q_opt), 'FontSize', 14);
legend({'训练数据', '测试数据', '预测值', '95%置信区间'}, 'Location', 'best');
grid on;
hold off;
    