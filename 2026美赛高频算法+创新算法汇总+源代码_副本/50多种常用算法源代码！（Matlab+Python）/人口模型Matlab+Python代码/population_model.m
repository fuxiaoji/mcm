% 人口模型案例说明：
% 本案例实现经典的Logistic人口增长模型，也称为S型增长模型。该模型考虑了环境承载能力，
% 克服了Malthus模型（指数增长）未考虑资源限制的缺陷。
%
% 模型公式：
% dP/dt = r*P*(1 - P/K)
%
% 其中：
% - P(t) 表示t时刻的人口数量
% - r 表示固有增长率（出生率减去死亡率）
% - K 表示环境承载能力（环境所能容纳的最大人口数量）
%
% 模型特点：
% - 当人口数量远小于K时，增长近似指数增长
% - 当人口数量接近K时，增长速度逐渐减慢
% - 最终人口数量将稳定在K附近
%
% 本案例通过数值方法求解微分方程，并与美国实际人口数据进行对比验证。

% 1. 定义Logistic人口增长模型
function dPdt = logistic_model(t, P, r, K)
    % Logistic人口增长模型微分方程
    % dP/dt = r*P*(1 - P/K)
    dPdt = r * P * (1 - P / K);
end

% 2. 设置模型参数
r = 0.025;       % 固有增长率（每年）
K = 300e6;       % 环境承载能力（3亿人）
P0 = 3.9e6;      % 初始人口（1790年美国人口）
t_span = [1790, 2050];  % 时间跨度（年）
t_eval = linspace(1790, 2050, 261);  % 评估时间点

% 3. 求解微分方程
% 使用匿名函数包装模型，固定r和K参数
model = @(t, P) logistic_model(t, P, r, K);
[t, P] = ode45(model, t_span, P0);

% 4. 美国实际人口数据（1790-2020年，单位：百万人）
years_actual = [1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, ...
                1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020];
populations_actual = [3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4, 38.6, 50.2, 62.9, 76.0, ...
                      92.0, 105.7, 122.8, 131.7, 150.7, 179.3, 203.2, 226.5, 248.7, 281.4, 308.7, 331.9];

% 5. 可视化结果
figure('Position', [100 100 800 500]);

% 绘制模型预测结果
plot(t, P/1e6, 'b-', 'LineWidth', 2);
hold on;

% 绘制实际人口数据
scatter(years_actual, populations_actual, 50, 'r', 'filled', 'MarkerEdgeColor', 'k');

% 添加标签和标题
xlabel('年份', 'FontSize', 12);
ylabel('人口数量（百万）', 'FontSize', 12);
title('Logistic人口增长模型与实际人口对比', 'FontSize', 14);
grid on;
legend({sprintf('Logistic模型预测 (r=%.3f, K=%.0f百万)', r, K/1e6), '美国实际人口数据'}, ...
       'FontSize', 10, 'Location', 'best');
xlim([1780, 2060]);
ylim([0, 350]);

% 添加模型公式说明
text(1800, 300, '\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)', ...
     'FontSize', 16, 'Interpreter', 'latex', ...
     'BackgroundColor', 'w', 'EdgeColor', 'none');

hold off;

% 6. 预测未来人口
future_years = [2030, 2040, 2050];
% 查找未来年份在求解结果中的索引
future_indices = interp1(t, 1:length(t), future_years, 'nearest');
future_populations = P(future_indices)/1e6;

fprintf('\n未来人口预测：\n');
for i = 1:length(future_years)
    fprintf('%d年: %.2f 百万人\n', future_years(i), future_populations(i));
end
    