% 数据包络法(DEA)案例说明：
% 本案例对6家制造企业的生产效率进行评价，考虑2项投入和2项产出：
% 投入指标：1. 固定资产(万元)  2. 员工人数(人)
% 产出指标：1. 总产值(万元)    2. 净利润(万元)
% 通过DEA-CCR模型（投入导向）计算各企业的效率值，判断是否DEA有效，
% 效率值为1且松弛变量为0的企业为DEA有效。

% 投入矩阵（6家企业×2项投入）
% 行：企业1-6，列：固定资产、员工人数
X = [
    500, 80;    % 企业1
    600, 90;    % 企业2
    400, 70;    % 企业3
    700, 100;   % 企业4
    300, 60;    % 企业5
    550, 85];   % 企业6

% 产出矩阵（6家企业×2项产出）
% 行：企业1-6，列：总产值、净利润
Y = [
    1200, 300;  % 企业1
    1350, 320;  % 企业2
    1000, 280;  % 企业3
    1500, 350;  % 企业4
    800, 220;   % 企业5
    1250, 310]; % 企业6

[n, m] = size(X);  % n=决策单元数, m=投入指标数
s = size(Y, 2);    % s=产出指标数

% 初始化结果变量
theta = zeros(n, 1);         % 效率值
lambda_ = zeros(n, n);       % 权重向量
s_minus = zeros(n, m);       % 投入松弛变量
s_plus = zeros(n, s);        % 产出松弛变量

% 对每个决策单元求解DEA模型
for j = 1:n
    % 变量总数: n个lambda + m个s⁻ + s个s⁺
    num_vars = n + m + s;
    
    % 目标函数系数：min θ + ε*(sum(s⁻)+sum(s⁺))
    f = zeros(num_vars, 1);
    f(n+1:n+m) = 1;    % s⁻的系数
    f(n+m+1:end) = 1;  % s⁺的系数
    
    % 约束矩阵
    Aeq = zeros(m + s, num_vars);
    beq = zeros(m + s, 1);
    
    % 投入约束: X*λ + s⁻ = θ*Xj
    Aeq(1:m, 1:n) = X';
    Aeq(1:m, n+1:n+m) = eye(m);
    beq(1:m) = X(j, :)';
    
    % 产出约束: Y*λ - s⁺ = Yj
    Aeq(m+1:m+s, 1:n) = -Y';
    Aeq(m+1:m+s, n+m+1:end) = -eye(s);
    beq(m+1:m+s) = -Y(j, :)';
    
    % 变量下界：λ ≥ 0, s⁻ ≥ 0, s⁺ ≥ 0
    lb = zeros(num_vars, 1);
    ub = [];
    
    % 求解线性规划
    options = optimoptions('linprog', 'Display', 'off');
    [x, fval] = linprog(f, [], [], Aeq, beq, lb, ub, options);
    
    % 提取结果
    theta(j) = fval;
    lambda_(j, :) = x(1:n)';
    s_minus(j, :) = x(n+1:n+m)';
    s_plus(j, :) = x(n+m+1:end)';
end

% 判断是否DEA有效（效率值=1且松弛变量全为0）
is_efficient = (theta >= 1 - 1e-6) & ...
               (sum(s_minus, 2) < 1e-6) & ...
               (sum(s_plus, 2) < 1e-6);

% 输出结果
disp("数据包络法(DEA-CCR模型)计算结果：");
fprintf("决策单元数: %d, 投入指标数: %d, 产出指标数: %d\n", n, m, s);
disp("\n各企业效率值及DEA有效性：");
for i = 1:n
    fprintf("企业%d: 效率值=%.4f, ", i, theta(i));
    if is_efficient(i)
        disp("DEA有效");
    else
        disp("DEA无效");
        fprintf("  投入松弛变量: "); disp(round(s_minus(i, :), 4));
        fprintf("  产出松弛变量: "); disp(round(s_plus(i, :), 4));
    end
end
