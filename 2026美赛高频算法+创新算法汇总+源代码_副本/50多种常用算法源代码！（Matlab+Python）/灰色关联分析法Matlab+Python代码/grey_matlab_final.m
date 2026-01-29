% 灰色关联分析法案例说明：
% 本案例对5个地区的经济发展水平进行评价，考虑4项指标：
% 1. GDP增长率(%)  2. 人均收入(万元)  3. 就业率(%)  4. 财政收入(亿元)
% 通过灰色关联分析，将各地区与理想参考序列（各项指标均为最优值）进行比较，
% 计算关联度并排序，判断各地区经济发展水平的优劣。

% 原始数据矩阵（5个地区×4个指标）
% 行：地区1-5，列：GDP增长率、人均收入、就业率、财政收入
data = [
    1.2, 3.4, 2.5, 5.6;   % 地区1
    2.1, 4.2, 3.1, 6.2;   % 地区2
    1.5, 2.8, 2.9, 4.8;   % 地区3
    3.0, 3.9, 4.0, 5.9;   % 地区4
    2.4, 3.5, 3.6, 6.0];  % 地区5

% 参考序列（理想方案）- 每个指标的最优值（此处取各指标最大值）
reference = [3.0, 4.2, 4.0, 6.2];

[m, n] = size(data);  % m=样本数, n=指标数

% 数据归一化（区间化法）
% 合并数据与参考序列以便统一归一化
combined = [data; reference];
max_val = max(combined);  % 各指标最大值
min_val = min(combined);  % 各指标最小值

% 初始化归一化后的数据
data_norm = zeros(m, n);
ref_norm = zeros(1, n);

for j = 1:n
    % 防止分母为0
    if max_val(j) ~= min_val(j)
        % 归一化到[0,1]区间
        data_norm(:, j) = (data(:, j) - min_val(j)) / (max_val(j) - min_val(j));
        ref_norm(j) = (reference(j) - min_val(j)) / (max_val(j) - min_val(j));
    else
        data_norm(:, j) = zeros(m, 1);
        ref_norm(j) = 0;
    end
end

% 计算绝对差
delta = abs(data_norm - repmat(ref_norm, m, 1));
max_delta = max(delta(:));  % 最大绝对差
min_delta = min(delta(:));  % 最小绝对差

% 计算关联系数（分辨系数rho通常取0.5）
rho = 0.5;
gamma = (min_delta + rho * max_delta) ./ (delta + rho * max_delta);

% 计算关联度（各指标关联系数的平均值）
r = mean(gamma, 2);

% 排序（从大到小）
[~, rank_idx] = sort(-r);
rank = zeros(m, 1);
for i = 1:m
    rank(rank_idx(i)) = i;  % 排名从1开始
end

% 输出结果
disp("灰色关联分析法计算结果：");
fprintf("参考序列（理想方案）: "); disp(reference);
disp("各地区与理想方案的关联度:");
disp(round(r, 4));
disp("各地区排名（从优到劣）:");
for i = 1:m
    fprintf("第%d名: 地区%d, 关联度: %.4f\n", i, rank_idx(i), r(rank_idx(i)));
end
