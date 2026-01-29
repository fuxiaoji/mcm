% 0-1规划模型案例说明：
% 本案例使用0-1规划解决设备选址问题。有4个候选地点可建立仓库，
% 每个地点的建设成本和覆盖需求不同，总预算限制为500万元。
% 目标是最大化总覆盖需求，且每个地点只能选择建设或不建设（0-1变量）。
% 0-1规划是整数规划的特例，决策变量只能取0或1。

% 目标函数：最大化总覆盖需求 = 15x1 + 12x2 + 20x3 + 10x4
% 转化为最小化问题：min -15x1 -12x2 -20x3 -10x4
f = [-15; -12; -20; -10];

% 不等式约束：A*x <= b
A = [200, 180, 250, 150];  % 成本约束：总建设成本不超过500万元
b = 500;

% 等式约束：无
Aeq = [];
beq = [];

% 变量类型：整数变量索引
intcon = 1:4;

% 变量下界：0
lb = zeros(4, 1);

% 变量上界：1（0-1变量）
ub = ones(4, 1);

% 求解0-1规划
[x, fval] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub);

% 输出结果
fprintf('求解状态: 成功\n');
fprintf('\n最优选址方案:\n');
total_cost = 0;
total_demand = 0;
for i = 1:4
    if x(i) == 1
        location_data = [200, 180, 250, 150; 15, 12, 20, 10];
        fprintf('在地点%d建设仓库: 成本%d万元, 覆盖需求%d千户\n', i-1, location_data(1,i), location_data(2,i));
        total_cost = total_cost + location_data(1,i);
        total_demand = total_demand + location_data(2,i);
    end
end

fprintf('\n总建设成本: %d万元, 总覆盖需求: %d千户\n', total_cost, total_demand);
fprintf('预算剩余: %d万元\n', 500 - total_cost);
