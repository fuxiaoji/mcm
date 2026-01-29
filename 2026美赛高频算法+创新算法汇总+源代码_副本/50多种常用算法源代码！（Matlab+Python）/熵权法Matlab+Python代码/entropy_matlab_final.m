% 熵权法案例说明：
% 本案例对5家企业的3项经济效益指标进行评价，指标分别为：
% 1. 净利润(万元)  2. 资产负债率(%)  3. 销售收入增长率(%)
% 所有指标均为效益型指标（数值越大越好），通过熵权法计算各指标的客观权重，
% 并对5家企业进行综合评分排序。

% 原始数据矩阵（5个样本×3个指标）
% 行：5家企业，列：净利润、资产负债率、销售收入增长率
data = [
    89, 90, 92;   % 企业1
    92, 88, 90;   % 企业2
    78, 95, 86;   % 企业3
    90, 85, 94;   % 企业4
    85, 92, 88];  % 企业5

[m, n] = size(data);  % m=样本数, n=指标数

% 数据归一化（正向化处理）
data_norm = zeros(m, n);
for j = 1:n
    max_val = max(data(:, j));  % 第j个指标的最大值
    min_val = min(data(:, j));  % 第j个指标的最小值
    % 防止分母为0
    if max_val ~= min_val
        % 归一化到[0,1]区间
        data_norm(:, j) = (data(:, j) - min_val) / (max_val - min_val);
    else
        data_norm(:, j) = zeros(m, 1);  % 若指标值相同，归一化为0
    end
end

% 计算第j项指标下第i个样本的比重p_ij
p = zeros(m, n);
for j = 1:n
    sum_col = sum(data_norm(:, j));  % 第j个指标归一化后总和
    if sum_col ~= 0
        p(:, j) = data_norm(:, j) / sum_col;  % 计算比重
    else
        p(:, j) = ones(m, 1) / m;  % 若总和为0，平均分配比重
    end
end

% 计算熵值e_j
e = zeros(1, n);
% 计算常数k（1/ln(m)）
if m > 1
    k = 1 / log(m);
else
    k = 0;
end
for j = 1:n
    % 计算每个指标的熵值，添加极小值防止log(0)
    e(j) = -k * sum(p(:, j) .* log(p(:, j) + eps));
end

% 计算信息熵冗余度d_j和权重
d = 1 - e;               % 冗余度 = 1 - 熵值
weights = d / sum(d);    % 权重归一化

% 计算各样本的综合得分
scores = data_norm * weights';

% 输出结果
disp("熵权法计算结果：");
fprintf("样本数: %d, 指标数: %d\n", m, n);
disp("各指标权重:");
disp(round(weights, 4));
disp("各样本综合得分及排名：");
% 排序并输出
[sorted_scores, sorted_indices] = sort(scores, 'descend');
for i = 1:m
    fprintf("第%d名: 样本%d, 得分: %.4f\n", i, sorted_indices(i), sorted_scores(i));
end
