% 最速下降法案例说明：
% 本案例使用最速下降法（梯度下降法）求解无约束优化问题。
% 目标函数为 f(x,y) = x² + 3y² - 2xy，这是一个凸函数，存在唯一最小值。
% 最速下降法是一种一阶优化算法，沿着目标函数梯度的反方向（最速下降方向）搜索最优解。

% 参数设置
initial_point = [5.0; 5.0];  % 初始点
learning_rate = 0.1;         % 学习率（步长）
max_iterations = 1000;       % 最大迭代次数
tolerance = 1e-6;            % 收敛容差

% 初始化
x = initial_point;
path = x;  % 记录迭代路径
iterations = 0;

% 最速下降法主循环
while iterations < max_iterations
    iterations = iterations + 1;
    
    % 计算梯度
    grad = gradient_func(x);
    
    % 判断收敛条件：梯度的模长小于 tolerance
    if norm(grad) < tolerance
        break;
    end
    
    % 沿负梯度方向更新（最速下降方向）
    x = x - learning_rate * grad;
    path = [path, x];  % 记录路径
end

% 计算最优值
optimal_value = objective_func(x);

% 输出结果
fprintf('最速下降法求解结果：\n');
fprintf('初始点: [%f, %f]\n', initial_point(1), initial_point(2));
fprintf('迭代次数: %d\n', iterations);
fprintf('最优解: x = %.6f, y = %.6f\n', x(1), x(2));
fprintf('最优值: f(x,y) = %.6f\n', optimal_value);
fprintf('最终梯度模长: %.6e\n', norm(gradient_func(x)));

% 可视化迭代过程（可选）
figure;
% 创建网格
[x_grid, y_grid] = meshgrid(-1:0.1:6, -1:0.1:6);
[rows, cols] = size(x_grid);
z_grid = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        z_grid(i,j) = objective_func([x_grid(i,j); y_grid(i,j)]);
    end
end

% 绘制等高线和迭代路径
contour(x_grid, y_grid, z_grid, 30);
colorbar;
hold on;

% 绘制迭代路径
plot(path(1,:), path(2,:), 'ro-', 'MarkerSize', 5);
plot(initial_point(1), initial_point(2), 'go', 'MarkerSize', 8);
plot(x(1), x(2), 'bo', 'MarkerSize', 8);

xlabel('x');
ylabel('y');
title('最速下降法迭代路径');
legend('迭代路径', '初始点', '最优解');
grid on;
hold off;

% 目标函数: f(x,y) = x² + 3y² - 2xy
function f = objective_func(x)
    f = x(1)^2 + 3*x(2)^2 - 2*x(1)*x(2);
end

% 目标函数的梯度: ∇f(x,y) = [2x-2y, 6y-2x]
function grad = gradient_func(x)
    grad = [
        2*x(1) - 2*x(2);  % 对x的偏导数
        6*x(2) - 2*x(1)   % 对y的偏导数
    ];
end
    