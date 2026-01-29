% 遗传算法案例说明：
% 本案例使用遗传算法求解Griewank函数的最小值。
% Griewank函数是一个具有大量局部最优解的多峰函数，全局最小值为0，位于(0,0,...,0)。
% 遗传算法模拟生物进化过程，通过选择、交叉和变异操作寻找最优解，
% 适合求解复杂的全局优化问题。

% 参数设置
bounds = [-600, 600;    % 变量范围
          -600, 600];
pop_size = 50;          % 种群大小
num_generations = 100;  % 进化代数

dim = size(bounds, 1);  % 问题维度

% 初始化种群
population = initialize_population(pop_size, dim, bounds);

% 评估初始种群
fitness = arrayfun(@(i) griewank(population(i, :)'), pop_size);

% 记录最优解
[best_value, best_idx] = min(fitness);
best_solution = population(best_idx, :)';
best_history = zeros(1, num_generations + 1);
best_history(1) = best_value;

% 主循环
for gen = 1:num_generations
    % 选择父代
    num_parents = pop_size // 2;
    parents = select(population, fitness, num_parents);
    
    % 交叉产生子代
    offspring_size = [pop_size - num_parents, dim];
    offspring = crossover(parents, offspring_size);
    
    % 变异
    mutation_rate = 0.1;  % 变异概率
    offspring = mutate(offspring, bounds, mutation_rate);
    
    % 形成新种群
    population = [parents; offspring];
    
    % 评估新种群
    fitness = arrayfun(@(i) griewank(population(i, :)'), pop_size);
    
    % 更新最优解
    [current_best_value, current_best_idx] = min(fitness);
    if current_best_value < best_value
        best_solution = population(current_best_idx, :)';
        best_value = current_best_value;
    end
    
    % 记录历史
    best_history(gen + 1) = best_value;
end

% 输出结果
fprintf('遗传算法求解Griewank函数结果：\n');
fprintf('最优解: x = %.6f, y = %.6f\n', best_solution(1), best_solution(2));
fprintf('最优值: f(x,y) = %.6f\n', best_value);
fprintf('理论最优解: (0, 0)，最优值: 0\n');

% 可视化结果
figure('Position', [100, 100, 1000, 500]);

% 3D曲面图
subplot(1, 2, 1);
[x, y] = meshgrid(linspace(bounds(1,1), bounds(1,2), 50), ...
                  linspace(bounds(2,1), bounds(2,2), 50));
z = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        z(i,j) = griewank([x(i,j); y(i,j)]);
    end
end
surf(x, y, z, 'EdgeColor', 'none');
colormap(viridis);
hold on;
scatter3(best_solution(1), best_solution(2), best_value, 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
title('Griewank函数曲面与最优解');
legend('函数曲面', '最优解');
hold off;

% 收敛曲线图
subplot(1, 2, 2);
plot(best_history);
xlabel('进化代数');
ylabel('最优值');
title('收敛曲线');
grid on;

% 初始化种群
function population = initialize_population(pop_size, dim, bounds)
    population = zeros(pop_size, dim);
    for i = 1:pop_size
        for j = 1:dim
            population(i, j) = bounds(j, 1) + rand() * (bounds(j, 2) - bounds(j, 1));
        end
    end
end

% 选择操作（轮盘赌选择）
function parents = select(population, fitness, num_parents)
    % 适应度越小越好，转换为选择概率（取倒数）
    fitness = max(fitness) - fitness + 1e-10;  % 确保非负
    probabilities = fitness / sum(fitness);
    
    % 选择父代
    parents = zeros(num_parents, size(population, 2));
    for i = 1:num_parents
        idx = randsample(length(fitness), 1, true, probabilities);
        parents(i, :) = population(idx, :);
    end
end

% 交叉操作（单点交叉）
function offspring = crossover(parents, offspring_size)
    offspring = zeros(offspring_size);
    crossover_rate = 0.8;  % 交叉概率
    
    for i = 1:offspring_size(1)
        % 随机选择两个父代
        parent1_idx = mod(i, size(parents, 1)) + 1;
        parent2_idx = mod(i + 1, size(parents, 1)) + 1;
        parent1 = parents(parent1_idx, :);
        parent2 = parents(parent2_idx, :);
        
        % 以一定概率进行交叉
        if rand() < crossover_rate
            % 随机选择交叉点
            crossover_point = randi(offspring_size(2) - 1);
            % 生成子代
            offspring(i, :) = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
        else
            % 不交叉，直接复制父代
            offspring(i, :) = parent1;
        end
    end
end

% 变异操作
function offspring = mutate(offspring, bounds, mutation_rate)
    [rows, cols] = size(offspring);
    for i = 1:rows
        for j = 1:cols
            % 以一定概率进行变异
            if rand() < mutation_rate
                % 高斯变异
                offspring(i, j) = offspring(i, j) + normrnd(0, 0.5);
                % 边界处理
                offspring(i, j) = max(bounds(j, 1), min(offspring(i, j), bounds(j, 2)));
            end
        end
    end
end

% Griewank函数：一个复杂的多峰函数
function f = griewank(x)
    n = length(x);
    sum_part = sum(x.^2) / 4000;
    prod_part = 1;
    for i = 1:n
        prod_part = prod_part * cos(x(i) / sqrt(i));
    end
    f = sum_part - prod_part + 1;
end
    