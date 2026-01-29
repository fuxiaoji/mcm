% 整数规划模型案例说明：
% 本案例使用整数规划解决投资组合问题。有5个投资项目可选，
% 每个项目有初始投资和预期收益，总预算限制为300万元。
% 目标是最大化总预期收益，且每个项目只能选择投资或不投资（0-1整数变量）。
% 整数规划适用于决策变量需取整数值的优化问题。

% 目标函数：最大化总预期收益 = 150x1 + 180x2 + 120x3 + 220x4 + 130x5
% 转化为最小化问题：min -150x1 -180x2 -120x3 -220x4 -130x5
f = [-150; -180; -120; -220; -130];

% 不等式约束：A*x <= b
A = [100, 120, 80, 150, 90];  % 投资约束：总投资不超过300万元
b = 300;

% 等式约束：无
Aeq = [];
beq = [];

% 变量类型：整数变量索引
intcon = 1:5;

% 变量下界：0
lb = zeros(5, 1);

% 变量上界：1（0-1变量）
ub = ones(5, 1);

% 求解整数规划
[x, fval] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub);

% 输出结果
fprintf('求解状态: 成功\n');
fprintf('\n最优投资组合:\n');
total_invest = 0;
total_profit = 0;
for i = 1:5
    if x(i) == 1
        project_data = [100, 120, 80, 150, 90; 150, 180, 120, 220, 130];
        fprintf('投资项目%d: 投资%d万元, 预期收益%d万元\n', i-1, project_data(1,i), project_data(2,i));
        total_invest = total_invest + project_data(1,i);
        total_profit = total_profit + project_data(2,i);
    end
end

fprintf('\n总投资: %d万元, 总预期收益: %d万元\n', total_invest, total_profit);
fprintf('预算剩余: %d万元\n', 300 - total_invest);
