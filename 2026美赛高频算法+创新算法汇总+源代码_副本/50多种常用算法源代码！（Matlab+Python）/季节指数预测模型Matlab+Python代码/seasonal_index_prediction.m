% 季节指数预测模型案例说明：
% 本案例使用季节指数法预测某景区未来4个季度的游客数量，
% 已知过去4年（16个季度）的游客数据（单位：千人）。
% 季节指数法适用于具有明显季节性波动的数据，通过分离
% 季节因素和趋势因素，先预测趋势，再用季节指数调整，
% 得到最终预测结果。

% 1. 准备数据
% 过去4年（16个季度）的游客数量（单位：千人）
% 数据呈现季节性：夏季（第2季度）和秋季（第3季度）为旺季
visitors = [
    32, 65, 58, 28,  % 第1年
    35, 70, 62, 30,  % 第2年
    38, 75, 68, 33,  % 第3年
    42, 80, 72, 36]; % 第4年
n_years = 4;       % 年数
n_seasons = 4;     % 每年的季节数（季度）
n = length(visitors);  % 数据总长度

% 2. 计算各季节的平均值
seasonal_data = reshape(visitors, n_seasons, n_years)';  % 按年和季节整理数据（行=年，列=季节）
seasonal_mean = mean(seasonal_data);  % 各季节的平均值

% 3. 计算总平均值
total_mean = mean(visitors);  % 所有数据的总平均值

% 4. 计算季节指数（季节平均值/总平均值）
seasonal_index = seasonal_mean / total_mean;

% 5. 计算 deseasonalized 数据（消除季节影响的数据）
deseasonalized = zeros(1, n);
for i = 1:n
    season = mod(i-1, n_seasons) + 1;  % 当前数据点所属的季节（1-based）
    deseasonalized(i) = visitors(i) / seasonal_index(season);
end

% 6. 拟合趋势线（线性趋势）
t = 1:n;  % 时间变量（1到n）
% 线性回归：y = a + b*t
p = polyfit(t, deseasonalized, 1);  % 计算回归系数，p(1)=b, p(2)=a
b = p(1);  % 斜率
a = p(2);  % 截距

% 7. 预测未来趋势值
forecast_num = 4;  % 预测未来4个季度（1年）
future_t = n+1:n+forecast_num;  % 未来时间点
future_trend = a + b * future_t;  % 未来趋势预测值

% 8. 用季节指数调整趋势预测值，得到最终预测值
future_seasons = mod((n-1):(n-1+forecast_num), n_seasons) + 1;
future_visitors = future_trend .* seasonal_index(future_seasons);

% 9. 计算历史数据拟合值
historical_trend = a + b * t;  % 历史趋势值
historical_fitted = zeros(1, n);
for i = 1:n
    season = mod(i-1, n_seasons) + 1;
    historical_fitted(i) = historical_trend(i) * seasonal_index(season);
end

% 10. 计算预测误差（均方根误差RMSE）
rmse = sqrt(mean((visitors - historical_fitted).^2));

% 11. 输出结果
disp("季节指数预测模型结果：");
fprintf("各季节指数: "); disp(round(seasonal_index, 4));
fprintf("趋势方程: y = %.4f + %.4f*t\n", a, b);
fprintf("均方根误差RMSE: %.2f\n", rmse);

disp("\n历史数据拟合结果：");
for i = 1:n_years
    for j = 1:n_seasons
        idx = (i-1)*n_seasons + j;
        fprintf("第%d年第%d季度: 实际值=%d, 拟合值=%.2f\n", ...
            i, j, visitors(idx), historical_fitted(idx));
    end
end

disp("\n未来4个季度预测结果：");
for i = 1:forecast_num
    fprintf("第%d年第%d季度预测值: %.2f千人\n", ...
        n_years+1, i, future_visitors(i));
end

% 12. 可视化结果
figure;
% 绘制历史数据
plot(1:n, visitors, 'bo-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '实际游客数量');
hold on;
% 绘制拟合数据
plot(1:n, historical_fitted, 'r--', 'LineWidth', 1.5, 'DisplayName', '拟合游客数量');
% 绘制预测数据
plot(n+1:n+forecast_num, future_visitors, 'g*-', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '预测游客数量');
% 添加网格线和标注
xticks(1:n_seasons:n+forecast_num);
xticklabels(arrayfun(@(x) sprintf('第%d年', x), 1:n_years+1, 'UniformOutput', false));
xlabel('时间');
ylabel('游客数量（千人）');
title('季节指数预测模型');
legend();
grid on;
hold off;
