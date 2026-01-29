% 战争模型案例说明：
% 本案例实现经典的兰彻斯特战争模型（Lanchester's model），该模型用于描述两个敌对部队在战斗中的兵力变化。
%
% 基本模型公式（平方律）：
% dx/dt = -b*y
% dy/dt = -a*x
%
% 其中：
% - x(t) 表示甲方在t时刻的兵力
% - y(t) 表示乙方在t时刻的兵力
% - a 表示乙方的战斗力系数（单位时间内每个乙方士兵消灭的甲方士兵数）
% - b 表示甲方的战斗力系数（单位时间内每个甲方士兵消灭的乙方士兵数）
%
% 模型特点：
% - 平方律模型适用于双方都能瞄准并攻击敌方任意目标的现代战争
% - 战斗力不仅取决于兵力数量，还取决于武器装备、训练水平等因素（体现在系数a和b中）
% - 当一方兵力减为0时，另一方获胜
%
% 本案例模拟两种不同初始条件下的战斗过程，展示兵力随时间的变化。

% 1. 定义兰彻斯特战争模型（平方律）
function dydt = lanchester_model(t, state, a, b)
    % 兰彻斯特战争模型微分方程组
    % dx/dt = -b*y
    % dy/dt = -a*x
    x = state(1);  % 甲方兵力
    y = state(2);  % 乙方兵力
    
    dydt = [-b * y;  % 甲方兵力变化率
            -a * x]; % 乙方兵力变化率
end

% 2. 设置模型参数和初始条件
% 情况1：甲方兵力占优
a1 = 0.05;       % 乙方战斗力系数
b1 = 0.08;       % 甲方战斗力系数
x0_1 = 100; y0_1 = 60;  % 初始兵力

% 情况2：乙方装备更精良（战斗力系数更高）
a2 = 0.12;       % 乙方战斗力系数（更高）
b2 = 0.08;       % 甲方战斗力系数
x0_2 = 100; y0_2 = 80;  % 初始兵力

t_span = [0, 15];  % 时间跨度（天）

% 3. 求解微分方程（两种情况）
% 情况1
model1 = @(t, state) lanchester_model(t, state, a1, b1);
[t1, state1] = ode45(model1, t_span, [x0_1; y0_1]);
x1 = state1(:, 1);  % 甲方兵力
y1 = state1(:, 2);  % 乙方兵力

% 情况2
model2 = @(t, state) lanchester_model(t, state, a2, b2);
[t2, state2] = ode45(model2, t_span, [x0_2; y0_2]);
x2 = state2(:, 1);  % 甲方兵力
y2 = state2(:, 2);  % 乙方兵力

% 4. 确定每种情况下的战斗结束时间
function [end_time, winner, remaining] = find_end_time(t, x, y)
    % 找到一方兵力接近0的时间点
    % 找到甲方兵力接近0的索引
    x_end_idx = find(x < 1, 1);
    % 找到乙方兵力接近0的索引
    y_end_idx = find(y < 1, 1);
    
    if isempty(x_end_idx)
        x_end_idx = length(x) + 1;  % 甲方未被消灭
    end
    if isempty(y_end_idx)
        y_end_idx = length(y) + 1;  % 乙方未被消灭
    end
    
    if x_end_idx < y_end_idx
        % 甲方先被消灭
        end_time = t(x_end_idx);
        winner = "乙方";
        remaining = y(x_end_idx);
    elseif y_end_idx < x_end_idx
        % 乙方先被消灭
        end_time = t(y_end_idx);
        winner = "甲方";
        remaining = x(y_end_idx);
    else
        % 同时被消灭
        end_time = t(end);
        winner = "双方";
        remaining = 0;
    end
end

% 计算情况1的结果
[end_time1, winner1, remaining1] = find_end_time(t1, x1, y1);

% 计算情况2的结果
[end_time2, winner2, remaining2] = find_end_time(t2, x2, y2);

% 5. 可视化结果
figure('Position', [100 100 1000 500]);

% 绘制情况1
subplot(1, 2, 1);
plot(t1, x1, 'b-', 'LineWidth', 2);
hold on;
plot(t1, y1, 'r-', 'LineWidth', 2);
xline(end_time1, 'k--', 'LineWidth', 1.5);
xlabel('时间（天）', 'FontSize', 12);
ylabel('兵力数量', 'FontSize', 12);
title(sprintf('情况1: %s获胜，剩余兵力: %.0f', winner1, remaining1), 'FontSize', 13);
grid on;
legend({sprintf('甲方兵力 (初始: %d)', x0_1), ...
        sprintf('乙方兵力 (初始: %d)', y0_1), ...
        sprintf('战斗结束: %.1f天', end_time1)}, ...
       'FontSize', 10, 'Location', 'best');
ylim([0, max(x0_1, y0_1) * 1.1]);
hold off;

% 绘制情况2
subplot(1, 2, 2);
plot(t2, x2, 'b-', 'LineWidth', 2);
hold on;
plot(t2, y2, 'r-', 'LineWidth', 2);
xline(end_time2, 'k--', 'LineWidth', 1.5);
xlabel('时间（天）', 'FontSize', 12);
ylabel('兵力数量', 'FontSize', 12);
title(sprintf('情况2: %s获胜，剩余兵力: %.0f', winner2, remaining2), 'FontSize', 13);
grid on;
legend({sprintf('甲方兵力 (初始: %d)', x0_2), ...
        sprintf('乙方兵力 (初始: %d)', y0_2), ...
        sprintf('战斗结束: %.1f天', end_time2)}, ...
       'FontSize', 10, 'Location', 'best');
ylim([0, max(x0_2, y0_2) * 1.1]);
hold off;

% 6. 输出战斗结果
fprintf("战斗结果摘要：\n");
fprintf("情况1: 甲方初始兵力=%d, 乙方初始兵力=%d, 乙方战斗力系数=%.2f, 甲方战斗力系数=%.2f\n", ...
        x0_1, y0_1, a1, b1);
fprintf("       获胜方: %s, 战斗持续时间: %.1f天, 剩余兵力: %.0f\n\n", ...
        winner1, end_time1, remaining1);

fprintf("情况2: 甲方初始兵力=%d, 乙方初始兵力=%d, 乙方战斗力系数=%.2f, 甲方战斗力系数=%.2f\n", ...
        x0_2, y0_2, a2, b2);
fprintf("       获胜方: %s, 战斗持续时间: %.1f天, 剩余兵力: %.0f\n", ...
        winner2, end_time2, remaining2);
    