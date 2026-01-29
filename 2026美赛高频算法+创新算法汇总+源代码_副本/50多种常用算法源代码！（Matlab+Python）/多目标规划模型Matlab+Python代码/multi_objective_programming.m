% 多目标规划模型案例说明：
% 本案例使用多目标规划解决工厂生产规划问题。工厂生产两种产品A和B，
% 有两个目标：1) 最大化利润；2) 最小化污染物排放。
% 多目标规划适用于存在多个需要同时优化的目标函数的问题，通常需要找到帕累托最优解。

% 初始猜测值
x0 = [10; 10];  % [产品A产量; 产品B产量]

% 变量边界：x1 >= 0, x2 >= 0
lb = [0; 0];
ub = [];

% 约束条件
A = [];
b = [];
Aeq = [];
beq = [];

% 尝试不同权重组合，探索帕累托最优前沿
weight_combinations = [
    0.1, 0.9;  % 更重视减排
    0.5, 0.5;  % 均衡考虑
    0.9, 0.1]; % 更重视利润

fprintf('多目标规划生产问题结果：\n');
fprintf('不同权重组合下的最优解：\n\n');

for i = 1:size(weight_combinations, 1)
    weights = weight_combinations(i, :);
    
    % 求解加权单目标问题
    options = optimoptions('fmincon', 'Display', 'off');
    [x, fval] = fmincon(@(x)weighted_objective(x, weights), x0, A, b, Aeq, beq, lb, ub, @constraints, options);
    
    x1 = x(1);
    x2 = x(2);
    
    % 计算各目标值
    [profit, emission] = objectives(x);
    
    % 计算约束使用情况
    material_used = 3*x1 + 5*x2;
    labor_used = 2*x1 + 3*x2;
    
    % 输出结果
    fprintf('权重组合 %d: 利润权重=%.1f, 减排权重=%.1f\n', i, weights(1), weights(2));
    fprintf('  最优解: 产品A=%.2f件, 产品B=%.2f件\n', x1, x2);
    fprintf('  目标值: 利润=%.2f元, 污染物排放=%.2f单位\n', profit, emission);
    fprintf('  约束使用: 原材料=%.2f/150, 工时=%.2f/90\n\n', material_used, labor_used);
end

% 目标函数：返回利润和排放量
function [profit, emission] = objectives(x)
    x1 = x(1);
    x2 = x(2);
    profit = 50*x1 + 80*x2;           % 利润
    emission = 0.2*x1^2 + 0.5*x2^2;  % 污染物排放
end

% 加权目标函数
function f = weighted_objective(x, weights)
    [profit, emission] = objectives(x);
    % 利润取负值是因为要最大化利润，而fmincon是求最小值
    f = weights(1)*(-profit) + weights(2)*emission;
end

% 约束条件函数
function [c, ceq] = constraints(x)
    % 不等式约束 c(x) <= 0
    c(1) = (3*x(1) + 5*x(2)) - 150;  % 原材料约束
    c(2) = (2*x(1) + 3*x(2)) - 90;   % 工时约束
    
    % 等式约束 ceq(x) = 0（无）
    ceq = [];
end
    