% 马尔可夫预测模型案例说明：
% 本案例使用马尔可夫链模型预测某股票价格的涨跌状态，
% 基于过去30天的价格状态记录。马尔可夫模型适用于预测
% 具有无后效性的随机过程，即未来状态只与当前状态有关，
% 与过去状态无关。本案例将股票价格分为3种状态：下跌(0)、
% 平稳(1)、上涨(2)，通过状态转移概率矩阵预测未来状态。

% 1. 准备数据
% 设置随机种子，确保结果可复现
rng(42);

% 生成30天的状态序列（模拟股票价格波动）
% 状态定义：0=下跌，1=平稳，2=上涨
states = [
    2, 2, 1, 0, 0, 1, 2, 2, 2, 1,
    0, 0, 1, 1, 2, 2, 1, 1, 0, 0,
    1, 2, 2, 1, 0, 1, 2, 2, 1, 0
];
n_days = length(states);  % 天数
n_states = 3;             % 状态数量：0-下跌，1-平稳，2-上涨
state_names = {'下跌', '平稳', '上涨'};

% 2. 计算状态转移概率矩阵
% 初始化转移计数矩阵
transition_counts = zeros(n_states, n_states);

% 统计状态转移次数
for i = 1:n_days - 1
    current_state = states(i) + 1;  % +1 转换为1-based索引
    next_state = states(i + 1) + 1;
    transition_counts(current_state, next_state) = transition_counts(current_state, next_state) + 1;
end

% 计算转移概率矩阵（行标准化）
transition_matrix = zeros(n_states, n_states);
for i = 1:n_states
    total = sum(transition_counts(i, :));
    if total > 0
        transition_matrix(i, :) = transition_counts(i, :) / total;
    end
end

% 3. 计算初始状态分布
initial_state = states(end) + 1;  % 最后一天的状态作为初始状态（1-based）
current_distribution = zeros(1, n_states);
current_distribution(initial_state) = 1.0;  % 初始分布：确定处于最后一个状态

% 4. 预测未来状态
forecast_days = 5;  % 预测未来5天
forecast_distributions = zeros(forecast_days, n_states);  % 存储每天的状态分布预测

% 迭代计算未来每一天的状态分布
for d = 1:forecast_days
    % 状态分布 = 当前分布 × 转移矩阵
    current_distribution = current_distribution * transition_matrix;
    forecast_distributions(d, :) = current_distribution;
end

% 确定每天最可能的状态
[~, most_likely_states] = max(forecast_distributions, [], 2);
most_likely_states = most_likely_states - 1;  % 转换回0-based

% 5. 计算各状态的稳态分布（长期预测）
% 通过多次迭代转移矩阵直到收敛
steady_state = current_distribution;
for i = 1:1000
    steady_state = steady_state * transition_matrix;
end

% 6. 输出结果
disp("马尔可夫预测模型结果：");
disp("\n状态转移概率矩阵：");
disp("行：当前状态，列：下一个状态");
fprintf("      下跌     平稳     上涨\n");
for i = 1:n_states
    fprintf("%s: %.4f, %.4f, %.4f\n", state_names{i}, ...
        transition_matrix(i, 1), transition_matrix(i, 2), transition_matrix(i, 3));
end

fprintf("\n最后一天的状态：%s\n", state_names{initial_state});

disp("\n未来5天的状态概率分布：");
for i = 1:forecast_days
    fprintf("第%d天:\n", i);
    for j = 1:n_states
        fprintf("  %s: %.2f%%\n", state_names{j}, forecast_distributions(i, j)*100);
    end
    fprintf("  最可能的状态: %s\n", state_names{most_likely_states(i)+1});
end

disp("\n长期稳态分布：");
for j = 1:n_states
    fprintf("  %s: %.2f%%\n", state_names{j}, steady_state(j)*100);
end

% 7. 可视化结果
figure;

% 绘制历史状态
subplot(2, 1, 1);
plot(1:n_days, states, 'bo-', 'MarkerSize', 6, 'LineWidth', 1.5);
set(gca, 'YTick', [0, 1, 2], 'YTickLabel', state_names);
xlabel('天数');
ylabel('状态');
title('股票价格历史状态');
grid on;

% 绘制预测概率
subplot(2, 1, 2);
days = n_days+1:n_days+forecast_days;
% 提取每种状态的预测概率
down_probs = forecast_distributions(:, 1);
stable_probs = forecast_distributions(:, 2);
up_probs = forecast_distributions(:, 3);

stackedplot = [down_probs', stable_probs', up_probs'];
area(days, stackedplot, 'FaceAlpha', 0.8);
hold on;
plot(days, most_likely_states, 'ko-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', '最可能状态');
legend(state_names, 'Location', 'best');
set(gca, 'YTick', 0:0.2:1);
xlabel('天数');
ylabel('概率');
title(sprintf('未来%d天状态预测概率分布', forecast_days));
grid on;
hold off;

tight_layout;
