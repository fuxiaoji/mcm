% 粒子群算法案例说明：
% 本案例使用粒子群优化算法(PSO)求解Rastrigin函数的最小值。
% Rastrigin函数是一个典型的非线性多峰函数，具有大量局部最优解，
% 常用于测试优化算法的性能。粒子群算法模拟鸟群觅食行为，
% 通过群体中个体间的协作和信息共享寻找最优解。

% 参数设置
dim = 2;                  % 问题维度
bounds = [-5.12, 5.12;    % 变量范围
          -5.12, 5.12];
num_particles = 30;       % 粒子数量
max_iter = 100;           % 最大迭代次数

% 初始化粒子群参数
w = 0.5;        % 惯性权重
c1 = 1;         % 认知系数
c2 = 2;         % 社会系数

% 初始化粒子位置和速度
particles = rand(num_particles, dim);  % 随机位置 [0,1)
% 将位置映射到给定范围
for i = 1:dim
    particles(:, i) = particles(:, i) * (bounds(i, 2) - bounds(i, 1)) + bounds(i, 1);
end

velocities = randn(num_particles, dim) * 0.1;  % 初始速度

% 初始化个体最优和全局最优
pbest_pos = particles;  % 个体最优位置
pbest_val = arrayfun(@(i) rastrigin(particles(i, :)'), num_particles);  % 个体最优值
[gbest_val, gbest_idx] = min(pbest_val);  % 全局最优值和索引
gbest_pos = pbest_pos(gbest_idx, :)';     % 全局最优位置

% 记录全局最优值的变化
gbest_history = zeros(1, max_iter + 1);
gbest_history(1) = gbest_val;

% 主循环
for iter = 1:max_iter
    % 更新速度和位置
    for i = 1:num_particles
        % 计算新速度
        r1 = rand(dim, 1);  % 随机因子1
        r2 = rand(dim, 1);  % 随机因子2
        cognitive = c1 * r1 .* (pbest_pos(i, :)' - particles(i, :)');  % 认知部分
        social = c2 * r2 .* (gbest_pos - particles(i, :)');            % 社会部分
        velocities(i, :)' = w * velocities(i, :)' + cognitive + social;  % 新速度
        
        % 更新位置
        particles(i, :)' = particles(i, :)' + velocities(i, :)';
        
        % 边界处理
        for j = 1:dim
            if particles(i, j) < bounds(j, 1)
                particles(i, j) = bounds(j, 1);
                velocities(i, j) = 0;  % 碰到边界速度归零
            elseif particles(i, j) > bounds(j, 2)
                particles(i, j) = bounds(j, 2);
                velocities(i, j) = 0;  % 碰到边界速度归零
            end
        end
    end
    
    % 评估当前位置
    current_val = arrayfun(@(i) rastrigin(particles(i, :)'), num_particles);
    
    % 更新个体最优
    improved = current_val < pbest_val;
    pbest_pos(improved, :) = particles(improved, :);
    pbest_val(improved) = current_val(improved);
    
    % 更新全局最优
    [current_gbest_val, current_gbest_idx] = min(pbest_val);
    if current_gbest_val < gbest_val
        gbest_pos = pbest_pos(current_gbest_idx, :)';
        gbest_val = current_gbest_val;
    end
    
    % 记录历史
    gbest_history(iter + 1) = gbest_val;
end

% 输出结果
fprintf('粒子群算法求解Rastrigin函数结果：\n');
fprintf('最优解: x = %.6f, y = %.6f\n', gbest_pos(1), gbest_pos(2));
fprintf('最优值: f(x,y) = %.6f\n', gbest_val);

% 可视化结果
figure('Position', [100, 100, 1000, 500]);

% 3D曲面图
subplot(1, 2, 1);
[x, y] = meshgrid(linspace(bounds(1,1), bounds(1,2), 100), ...
                  linspace(bounds(2,1), bounds(2,2), 100));
z = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        z(i,j) = rastrigin([x(i,j); y(i,j)]);
    end
end
surf(x, y, z, 'EdgeColor', 'none');
colormap(viridis);
hold on;
scatter3(gbest_pos(1), gbest_pos(2), gbest_val, 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
title('Rastrigin函数曲面与最优解');
legend('函数曲面', '最优解');
hold off;

% 收敛曲线图
subplot(1, 2, 2);
plot(gbest_history);
xlabel('迭代次数');
ylabel('全局最优值');
title('收敛曲线');
grid on;

% Rastrigin函数：一个复杂的多峰函数，最小值在(0,0)处，值为0
function f = rastrigin(x)
    A = 10;
    n = length(x);
    f = A * n + sum(x.^2 - A * cos(2 * pi * x));
end
    