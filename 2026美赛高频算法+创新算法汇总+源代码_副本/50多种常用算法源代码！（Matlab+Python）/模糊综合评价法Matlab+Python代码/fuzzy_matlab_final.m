% 模糊综合评价法案例说明：
% 本案例对5名员工的工作表现进行评价，考虑3项指标：
% 1. 工作效率(越高越好)  2. 团队协作(越高越好)  3. 创新能力(越高越好)
% 评价等级分为3级：优、中、差，通过模糊综合评价法计算各员工的隶属度，
% 并给出综合得分和排名。

% 原始数据矩阵（5个样本×3个指标）
% 行：员工1-5，列：工作效率、团队协作、创新能力
data = [
    85, 90, 88;   % 员工1
    92, 85, 90;   % 员工2
    78, 92, 86;   % 员工3
    88, 88, 94;   % 员工4
    90, 82, 89];  % 员工5

% 各指标权重
weights = [0.3, 0.4, 0.3];

% 评价等级标准（3个等级：优、中、差）
% 每行对应一个指标的三个等级标准
criteria = [
    90, 80, 70;  % 工作效率的优、中、差标准
    90, 80, 70;  % 团队协作的优、中、差标准
    90, 80, 70]; % 创新能力的优、中、差标准

[m, n] = size(data);  % m=样本数, n=指标数
k = size(criteria, 2);  % k=评价等级数

% 为每个指标创建单独的隶属度矩阵，避免三维索引问题
R1 = zeros(m, k);  % 指标1的隶属度矩阵
R2 = zeros(m, k);  % 指标2的隶属度矩阵
R3 = zeros(m, k);  % 指标3的隶属度矩阵

% 计算指标1的隶属度
for i = 1:m
    x = data(i, 1);
    s = criteria(1, :);
    
    % 优
    if x >= s(1)
        R1(i, 1) = 1;
    elseif x < s(2)
        R1(i, 1) = 0;
    else
        R1(i, 1) = (x - s(2)) / (s(1) - s(2));
    end
    
    % 中
    if x >= s(1) || x < s(3)
        R1(i, 2) = 0;
    elseif x >= s(2)
        R1(i, 2) = (s(1) - x) / (s(1) - s(2));
    else
        R1(i, 2) = (x - s(3)) / (s(2) - s(3));
    end
    
    % 差
    if x <= s(3)
        R1(i, 3) = 1;
    elseif x > s(2)
        R1(i, 3) = 0;
    else
        R1(i, 3) = (s(2) - x) / (s(2) - s(3));
    end
end

% 计算指标2的隶属度
for i = 1:m
    x = data(i, 2);
    s = criteria(2, :);
    
    % 优
    if x >= s(1)
        R2(i, 1) = 1;
    elseif x < s(2)
        R2(i, 1) = 0;
    else
        R2(i, 1) = (x - s(2)) / (s(1) - s(2));
    end
    
    % 中
    if x >= s(1) || x < s(3)
        R2(i, 2) = 0;
    elseif x >= s(2)
        R2(i, 2) = (s(1) - x) / (s(1) - s(2));
    else
        R2(i, 2) = (x - s(3)) / (s(2) - s(3));
    end
    
    % 差
    if x <= s(3)
        R2(i, 3) = 1;
    elseif x > s(2)
        R2(i, 3) = 0;
    else
        R2(i, 3) = (s(2) - x) / (s(2) - s(3));
    end
end

% 计算指标3的隶属度
for i = 1:m
    x = data(i, 3);
    s = criteria(3, :);
    
    % 优
    if x >= s(1)
        R3(i, 1) = 1;
    elseif x < s(2)
        R3(i, 1) = 0;
    else
        R3(i, 1) = (x - s(2)) / (s(1) - s(2));
    end
    
    % 中
    if x >= s(1) || x < s(3)
        R3(i, 2) = 0;
    elseif x >= s(2)
        R3(i, 2) = (s(1) - x) / (s(1) - s(2));
    else
        R3(i, 2) = (x - s(3)) / (s(2) - s(3));
    end
    
    % 差
    if x <= s(3)
        R3(i, 3) = 1;
    elseif x > s(2)
        R3(i, 3) = 0;
    else
        R3(i, 3) = (s(2) - x) / (s(2) - s(3));
    end
end

% 计算综合评价矩阵（加权求和）
evaluation = zeros(m, k);
for i = 1:m
    evaluation(i, 1) = weights(1)*R1(i,1) + weights(2)*R2(i,1) + weights(3)*R3(i,1);
    evaluation(i, 2) = weights(1)*R1(i,2) + weights(2)*R2(i,2) + weights(3)*R3(i,2);
    evaluation(i, 3) = weights(1)*R1(i,3) + weights(2)*R2(i,3) + weights(3)*R3(i,3);
end

% 计算综合得分（等级赋值：优=3, 中=2, 差=1）
level_scores = k:-1:1;  % 从高到低的得分
final_scores = evaluation * level_scores';

% 排序（从优到劣）
[sorted_scores, sorted_indices] = sort(final_scores, 'descend');

% 输出结果
disp("模糊综合评价法计算结果：");
disp("评价等级标准（优、中、差）：");
for j = 1:n
    fprintf("指标%d标准: ", j); disp(criteria(j, :));
end
disp("\n综合评价矩阵（每行一个样本，每列一个等级）：");
disp(round(evaluation, 4));
disp("\n各样本综合得分及排名：");
for i = 1:m
    fprintf("第%d名: 员工%d, 得分: %.4f\n", i, sorted_indices(i), sorted_scores(i));
end
