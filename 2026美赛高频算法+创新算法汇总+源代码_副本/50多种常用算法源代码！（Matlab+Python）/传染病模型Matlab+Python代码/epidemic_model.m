% 传染病模型案例说明：
% 本案例实现经典的SIR传染病模型，该模型将人群分为三类：
% - S (Susceptible)：易感人群，可能被感染的健康人群
% - I (Infected)：感染人群，已经感染并具有传染性的人群
% - R (Recovered)：康复人群，已经康复并获得免疫力的人群
%
% 模型公式：
% dS/dt = -β*S*I/N
% dI/dt = β*S*I/N - γ*I
% dR/dt = γ*I
%
% 其中：
% - N = S + I + R 表示总人口数（假设恒定）
% - β 表示感染率（每个感染者单位时间内传染的易感者数量）
% - γ 表示恢复率（单位时间内从感染者中康复的比例）
% - R0 = β/γ 表示基本再生数，衡量病毒传播能力
%
% 模型特点：
% - 描述了传染病在人群中的传播、发展和消退过程
% - R0 > 1 时，传染病会流行；R0 < 1 时，传染病会逐渐消失
% - 感染人群通常会先增加到一个峰值，然后逐渐减少
%
% 本案例模拟不同参数下的疫情发展曲线，并分析基本再生数R0对疫情的影响。

% 1. 定义SIR传染病模型
function dydt = sir_model(t, state, beta, gamma, N)
    % SIR传染病模型微分方程组
    S = state(1);  % 易感人群
    I = state(2);  % 感染人群
    R = state(3);  % 康复人群
    
    dydt = [-beta * S * I / N;    % 易感人群变化率
            beta * S * I / N - gamma * I;  % 感染人群变化率
            gamma * I];           % 康复人群变化率
end

% 2. 设置模型参数和初始条件
N = 100000;               % 总人口数
I0 = 10;                  % 初始感染人数
R0_initial = 0;           % 初始康复人数
S0 = N - I0 - R0_initial; % 初始易感人数

% 三种不同传播能力的场景
scenarios = [
    struct('beta', 0.3, 'gamma', 0.1, 'label', 'R0=3.0 (高传播性)'),  % R0=β/γ=3.0
    struct('beta', 0.15, 'gamma', 0.1, 'label', 'R0=1.5 (中传播性)'), % R0=1.5
    struct('beta', 0.08, 'gamma', 0.1, 'label', 'R0=0.8 (低传播性)')  % R0=0.8
];

t_span = [0, 100];  % 时间跨度（天）

% 3. 求解微分方程（三种场景）
solutions = cell(length(scenarios), 1);
for i = 1:length(scenarios)
    scenario = scenarios(i);
    model = @(t, state) sir_model(t, state, scenario.beta, scenario.gamma, N);
    [t, state] = ode45(model, t_span, [S0; I0; R0_initial]);
    solutions{i} = struct('t', t, 'S', state(:,1), 'I', state(:,2), 'R', state(:,3));
end

% 4. 分析关键指标
function result = analyze_scenario(solution, N, label)
    % 分析疫情的关键指标
    I = solution.I;
    % 感染人数峰值及出现时间
    [peak_I, peak_idx] = max(I);
    peak_time = solution.t(peak_idx);
    % 最终感染比例
    final_infected_ratio = (N - solution.S(end)) / N * 100;  % 初始易感者中最终被感染的比例
    
    result = struct( ...
        'label', label, ...
        'peak_time', peak_time, ...
        'peak_I', peak_I, ...
        'final_infected_ratio', final_infected_ratio ...
    );
end

% 分析所有场景
analysis_results = cell(length(scenarios), 1);
for i = 1:length(scenarios)
    analysis_results{i} = analyze_scenario(solutions{i}, N, scenarios(i).label);
end

% 5. 可视化结果
figure('Position', [100 100 900 800]);

% 绘制各人群随时间变化曲线（三种场景对比）
subplot(2, 1, 1);
colors = lines(3);  % 三种颜色
for i = 1:length(scenarios)
    sol = solutions{i};
    % 只对感染人群添加标签，避免图例过于复杂
    if i == 1
        plot(sol.t, sol.S, 'Color', colors(i,:), 'LineWidth', 1.5, 'Alpha', 0.7);
        hold on;
        plot(sol.t, sol.I, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', scenarios(i).label);
        plot(sol.t, sol.R, 'Color', colors(i,:), 'LineWidth', 1.5, 'Alpha', 0.7);
    else
        plot(sol.t, sol.S, 'Color', colors(i,:), 'LineWidth', 1.5, 'Alpha', 0.7);
        plot(sol.t, sol.I, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', scenarios(i).label);
        plot(sol.t, sol.R, 'Color', colors(i,:), 'LineWidth', 1.5, 'Alpha', 0.7);
    end
end

xlabel('时间（天）', 'FontSize', 12);
ylabel('人数', 'FontSize', 12);
title('不同传播能力下的SIR模型曲线', 'FontSize', 14);
grid on;
legend('Location', 'best', 'FontSize', 10);
text(5, N*0.9, 'S: 易感人群 (浅色)\nI: 感染人群 (深色)\nR: 康复人群 (中色)', ...
     'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'none');
hold off;

% 单独绘制感染人群曲线，突出峰值
subplot(2, 1, 2);
for i = 1:length(scenarios)
    sol = solutions{i};
    res = analysis_results{i};
    plot(sol.t, sol.I, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', scenarios(i).label);
    hold on;
    % 标记峰值点
    scatter(res.peak_time, res.peak_I, 50, colors(i,:), 'filled', 'MarkerEdgeColor', 'k');
    text(res.peak_time+1, res.peak_I, ...
         sprintf('峰值: %.0f人\n时间: %.1f天', res.peak_I, res.peak_time), ...
         'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'none');
end

xlabel('时间（天）', 'FontSize', 12);
ylabel('感染人数', 'FontSize', 12);
title('不同传播能力下的感染人群变化', 'FontSize', 14);
grid on;
legend('Location', 'best', 'FontSize', 10);
hold off;

% 6. 输出分析结果
fprintf("疫情分析结果：\n");
fprintf("总人口数: %d人, 初始感染人数: %d人\n", N, I0);
fprintf("%s\n", repmat('-', 1, 70));
fprintf("%-20s %-20s %-20s %-20s\n", "场景", "感染峰值时间(天)", "感染峰值人数", "最终感染比例(%)");
fprintf("%s\n", repmat('-', 1, 70));
for i = 1:length(analysis_results)
    res = analysis_results{i};
    fprintf("%-20s %-20.1f %-20.0f %-20.1f\n", ...
            res.label, res.peak_time, res.peak_I, res.final_infected_ratio);
end
    