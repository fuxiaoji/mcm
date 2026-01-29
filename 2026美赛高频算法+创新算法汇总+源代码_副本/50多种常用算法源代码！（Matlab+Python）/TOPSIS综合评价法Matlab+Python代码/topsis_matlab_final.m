% TOPSIS模型案例说明：
% 本案例对5款智能手机进行综合评价，考虑4项指标：
% 1. 性能评分(越高越好)  2. 价格(越低越好)  3. 续航时间(越长越好)  4. 拍照评分(越高越好)
% 通过TOPSIS法计算各款手机与理想方案和负理想方案的距离，进而得到综合评价指数，
% 最终对5款手机进行排序。

% 原始数据矩阵（5个样本×4个指标）
% 行：手机1-5，列：性能评分、价格、续航时间、拍照评分
data = [
    89, 3999, 48, 92;   % 手机1
    92, 4299, 45, 90;   % 手机2
    78, 2999, 50, 86;   % 手机3
    90, 4499, 42, 94;   % 手机4
    85, 3599, 46, 88];  % 手机5

% 各指标权重
weights = [0.3, 0.2, 0.25, 0.25];

% 指标类型（1表示效益型，0表示成本型）
% 性能、续航、拍照为效益型；价格为成本型
is_benefit = [1, 0, 1, 1];

[m, n] = size(data);  % m=样本数, n=指标数

% 数据归一化（标准化处理）
data_norm = zeros(m, n);
for j = 1:n
    % 计算第j个指标的模长
    norm = sqrt(sum(data(:, j) .^2));
    data_norm(:, j) = data(:, j) / norm;  % 归一化
end

% 加权归一化
weighted_data = data_norm .* repmat(weights, m, 1);  % 每个指标乘以相应权重

% 确定正理想解和负理想解
Z_plus = zeros(1, n);   % 正理想解（最优方案）
Z_minus = zeros(1, n);  % 负理想解（最劣方案）

for j = 1:n
    if is_benefit(j)  % 效益型指标：越大越好
        Z_plus(j) = max(weighted_data(:, j));  % 取最大值
        Z_minus(j) = min(weighted_data(:, j)); % 取最小值
    else  % 成本型指标：越小越好
        Z_plus(j) = min(weighted_data(:, j));  % 取最小值
        Z_minus(j) = max(weighted_data(:, j)); % 取最大值
    end
end

% 计算各样本到正理想解和负理想解的距离
D_plus = sqrt(sum((weighted_data - repmat(Z_plus, m, 1)) .^2, 2));  % 到正理想解的距离
D_minus = sqrt(sum((weighted_data - repmat(Z_minus, m, 1)) .^2, 2)); % 到负理想解的距离

% 计算综合评价指数（贴近度）
C = D_minus ./ (D_plus + D_minus);

% 排序（从优到劣）
[~, rank_idx] = sort(-C);
rank = zeros(m, 1);
for i = 1:m
    rank(rank_idx(i)) = i;  % 排名从1开始
end

% 输出结果
disp("TOPSIS模型计算结果：");
fprintf("正理想解(最优方案): "); disp(round(Z_plus, 4));
fprintf("负理想解(最劣方案): "); disp(round(Z_minus, 4));
disp("各样本综合评价指数:");
disp(round(C, 4));
disp("排序结果（从优到劣）:");
for i = 1:m
    fprintf("第%d名: 样本%d, 贴近度: %.4f\n", i, rank_idx(i), C(rank_idx(i)));
end
