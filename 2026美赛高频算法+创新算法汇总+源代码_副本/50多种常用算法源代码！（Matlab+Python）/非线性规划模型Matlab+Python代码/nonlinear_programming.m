% 非线性规划模型案例说明：
% 本案例使用非线性规划解决生产优化问题。某工厂生产两种产品A和B，
% 生产成本和销售价格存在非线性关系，目标是最大化净利润。
% 非线性规划适用于目标函数或约束条件包含非线性表达式的优化问题。

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

% 求解非线性规划
options = optimoptions('fmincon', 'Display', 'off');
[x, fval] = fmincon(@objective, x0, A, b, Aeq, beq, lb, ub, @constraints, options);

% 计算结果
x1 = x(1);
x2 = x(2);
revenue = (100 - 0.5*x1)*x1 + (150 - 0.8*x2)*x2;
cost = 500 + 20*x1 + 30*x2 + 0.1*x1^2 + 0.2*x2^2;
profit = revenue - cost;

% 输出结果
fprintf('非线性规划生产优化问题结果：\n');
fprintf('最优解: 产品A生产 %.2f 件, 产品B生产 %.2f 件\n', x1, x2);
fprintf('销售收入: %.2f 元\n', revenue);
fprintf('生产成本: %.2f 元\n', cost);
fprintf('最大净利润: %.2f 元\n', profit);

% 约束条件使用情况
material_used = 2*x1 + 3*x2;
labor_used = 5*x1 + 4*x2;
fprintf('\n约束条件使用情况:\n');
fprintf('原材料约束: 实际使用 %.2f, 限制 200\n', material_used);
fprintf('工时约束: 实际使用 %.2f, 限制 300\n', labor_used);

% 目标函数：最大化净利润 = 销售收入 - 生产成本
% 注意：返回负值，因为fmincon是求最小值
function f = objective(x)
    x1 = x(1);
    x2 = x(2);
    % 销售收入（非线性）
    revenue = (100 - 0.5*x1)*x1 + (150 - 0.8*x2)*x2;
    % 生产成本（非线性）
    cost = 500 + 20*x1 + 30*x2 + 0.1*x1^2 + 0.2*x2^2;
    f = -(revenue - cost);  % 最小化负净利润等价于最大化净利润
end

% 约束条件函数
function [c, ceq] = constraints(x)
    % 不等式约束 c(x) <= 0
    c(1) = (2*x(1) + 3*x(2)) - 200;  % 原材料约束
    c(2) = (5*x(1) + 4*x(2)) - 300;  % 工时约束
    
    % 等式约束 ceq(x) = 0（无）
    ceq = [];
end
    