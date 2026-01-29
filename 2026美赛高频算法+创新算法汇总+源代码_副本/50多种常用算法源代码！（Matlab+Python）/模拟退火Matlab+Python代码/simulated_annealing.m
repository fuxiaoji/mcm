% 模拟退火算法案例说明：
% 本案例使用模拟退火算法求解Schwefel函数的最小值。
% Schwefel函数是一个具有强多峰特性的复杂函数，全局最优值在(420.9687,420.9687)附近。
% 模拟退火算法灵感来源于物理退火过程，通过接受一定概率的劣解跳出局部最优，
% 适合求解复杂的全局优化问题。

% 参数设置
bounds = [-500, 500;    % 变量范围
          -500, 500];
initial_temp = 100.0;   % 初始温度
cooling_rate = 0.95;    % 冷却速率
max_iter = 1000;        % 最大迭代次数
step_size = 50.0;       % 初始步长

dim = size(bounds, 1);  % 问题维度

% 初始化当前解
current_solution = zeros(dim, 1);
for i = 1:dim
    current_solution(i) = bounds(i, 1) + rand() * (bounds(i, 2) - bounds(i, 1));
end
current_value = schwefel(current_solution);

% 初始化最优解
best_solution = current_solution;
best_value = current_value;

% 记录历史
current_temp = initial_temp;
best_history = zeros(1, max_iter + 1);
best_history(1) = best_value;

% 主循环
for i = 1:max_iter
    % 生成邻域解
    neighbor = generate_neighbor(current_solution, bounds, step_size);
    neighbor_value = schwefel(neighbor);
    
    % 计算能量差（目标函数差）
    delta = neighbor_value - current_value;
    
    % 接受准则：如果更优则接受，否则以一定概率接受
    if delta < 0 || rand() < exp(-delta / current_temp)
        current_solution = neighbor;
        current_value = neighbor_value;
        
        % 更新最优解
        if current_value < best_value
            best_solution = current_solution;
            best_value = current_value;
        end
    end
    
    % 降温
    current_temp = current_temp * cooling_rate;
    
    % 记录历史
    best_history(i + 1) = best_value;
    
    % 动态调整步长（可选）
    if mod(i, 100) == 0 && i > 0
        step_size = max(0.1, step_size * 0.95);
    end
end

% 输出结果
fprintf('模拟退火算法求解Schwefel函数结果：\n');
fprintf('最优解: x = %.6f, y = %.6f\n', best_solution(1), best_solution(2));
fprintf('最优值: f(x,y) = %.6f\n', best_value);
fprintf('理论最优解附近: (420.9687, 420.9687)\n');

% 可视化结果
figure('Position', [100, 100, 1000, 500]);

% 3D曲面图
subplot(1, 2, 1);
[x, y] = meshgrid(linspace(bounds(1,1), bounds(1,2), 50), ...
                  linspace(bounds(2,1), bounds(2,2), 50));
z = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        z(i,j) = schwefel([x(i,j); y(i,j)]);
    end
end
surf(x, y, z, 'EdgeColor', 'none');
colormap(viridis);
hold on;
scatter3(best_solution(1), best_solution(2), best_value, 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
title('Schwefel函数曲面与最优解');
legend('函数曲面', '最优解');
hold off;

% 收敛曲线图
subplot(1, 2, 2);
plot(best_history);
xlabel('迭代次数');
ylabel('最优值');
title('收敛曲线');
grid on;

% 生成邻域解
function neighbor = generate_neighbor(x, bounds, step_size)
    neighbor = x;
    dim = length(x);
    
    % 随机选择一个维度进行扰动
    idx = randi(dim);
    % 生成随机扰动
    neighbor(idx) = neighbor(idx) + (rand() * 2 - 1) * step_size;
    
    % 边界处理
    neighbor(idx) = max(bounds(idx, 1), min(neighbor(idx), bounds(idx, 2)));
end

% Schwefel函数：一个复杂的多峰函数
function f = schwefel(x)
    n = length(x);
    f = 418.9829 * n - sum(x .* sin(sqrt(abs(x))));
end
    