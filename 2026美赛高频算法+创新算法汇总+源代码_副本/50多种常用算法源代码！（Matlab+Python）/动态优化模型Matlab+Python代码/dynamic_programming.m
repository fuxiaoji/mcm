% 动态优化模型（动态规划）案例说明：
% 本案例使用动态规划解决背包问题。有5个物品，每个物品有重量和价值，
% 背包最大容量为10公斤。目标是在不超过背包容量的前提下，最大化总价值。
% 动态规划通过将复杂问题分解为子问题，利用子问题的解构建原问题的解。

% 物品数据：[重量(kg), 价值(元)]
items = [
    2, 6;   % 物品0
    2, 3;   % 物品1
    6, 5;   % 物品2
    5, 4;   % 物品3
    4, 6];  % 物品4

weights = items(:, 1);  % 重量向量
values = items(:, 2);   % 价值向量
capacity = 10;          % 背包容量(kg)
n = size(items, 1);     % 物品数量

% 创建二维数组dp，dp(i,j)表示考虑前i个物品，背包容量为j-1时的最大价值
dp = zeros(n + 1, capacity + 1);

% 填充dp数组
for i = 2:n + 1
    for j = 1:capacity + 1
        current_weight = weights(i - 1);  % 当前物品重量
        current_value = values(i - 1);    % 当前物品价值
        current_capacity = j - 1;         % 当前容量（j-1是实际容量值）
        
        if current_weight <= current_capacity
            % 可以放入第i个物品，比较放入和不放入的价值
            dp(i, j) = max(current_value + dp(i - 1, j - current_weight), dp(i - 1, j));
        else
            % 不能放入第i个物品，继承前i-1个物品的最佳价值
            dp(i, j) = dp(i - 1, j);
        end
    end
end

% 回溯找出选择的物品
selected_items = [];
j = capacity + 1;  % 对应容量为capacity
for i = n + 1:-1:2
    % 如果当前状态与不包含第i个物品的状态不同，说明选择了第i个物品
    if dp(i, j) ~= dp(i - 1, j)
        selected_items = [i - 2, selected_items];  % 转换为0-based索引
        j = j - weights(i - 1);  % 减去该物品的重量
    end
end

% 输出结果
fprintf('动态规划解决背包问题案例：\n');
fprintf('背包最大容量: %dkg\n', capacity);
fprintf('\n物品列表:\n');
for i = 1:n
    fprintf('物品%d: 重量%dkg, 价值%d元\n', i - 1, items(i, 1), items(i, 2));
end

max_value = dp(n + 1, capacity + 1);
fprintf('\n最大总价值: %d元\n', max_value);
fprintf('选择的物品:\n');
total_weight = 0;
for i = 1:length(selected_items)
    idx = selected_items(i) + 1;  % 转换为1-based索引
    fprintf('物品%d: 重量%dkg, 价值%d元\n', selected_items(i), items(idx, 1), items(idx, 2));
    total_weight = total_weight + items(idx, 1);
end

fprintf('\n总重量: %dkg, 剩余容量: %dkg\n', total_weight, capacity - total_weight);
