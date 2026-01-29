% 连续变量相关性分析案例说明：
% 本案例实现三种常用的连续变量相关性分析方法：
% 1. 协方差：衡量两个变量变化趋势的一致性，值为正表示同向变化，值为负表示反向变化
% 2. Pearson相关系数：标准化的协方差，取值范围[-1,1]，衡量线性相关程度
% 3. Spearman相关系数：基于变量秩次的非参数方法，取值范围[-1,1]，衡量单调相关关系
%
% 示例使用波士顿房价数据集（或替代数据集），分析房屋平均房间数与房屋价格之间的相关性，
% 并对比三种方法的结果差异和适用场景。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行完整分析');
end

% 加载数据集
disp('加载数据集...');
try
    % 尝试加载波士顿房价数据集
    data = load('boston.mat');
    boston = data.boston;
    x = boston(:,6);    % 平均房间数
    y = boston(:,14);   % 房屋价格
    var1_name = '平均房间数';
    var2_name = '房屋价格(千美元)';
catch
    % 加载替代数据集：鸢尾花数据集
    load fisheriris;
    x = meas(:,1);      % 花萼长度
    y = meas(:,3);      % 花瓣长度
    var1_name = '花萼长度(cm)';
    var2_name = '花瓣长度(cm)';
end

% 1. 计算协方差
mean_x = mean(x);
mean_y = mean(y);
covariance = mean((x - mean_x) .* (y - mean_y));
% 使用Matlab内置函数验证
cov_matrix = cov([x, y]);
matlab_cov = cov_matrix(1,2);

% 2. 计算Pearson相关系数
std_x = std(x, 0);    % 总体标准差
std_y = std(y, 0);
pearson_corr = covariance / (std_x * std_y);
% 使用Matlab函数验证并获取p值
[pearson_corr_matlab, pearson_p] = corrcoef(x, y);
pearson_corr_matlab = pearson_corr_matlab(1,2);

% 3. 计算Spearman相关系数（基于秩次）
x_rank = tiedrank(x);  % 计算秩次，处理平局
y_rank = tiedrank(y);
mean_rank_x = mean(x_rank);
mean_rank_y = mean(y_rank);
cov_rank = mean((x_rank - mean_rank_x) .* (y_rank - mean_rank_y));
std_rank_x = std(x_rank, 0);
std_rank_y = std(y_rank, 0);
spearman_corr = cov_rank / (std_rank_x * std_rank_y);
% 使用Matlab函数验证并获取p值
[spearman_corr_matlab, spearman_p] = corr(x, y, 'Type', 'Spearman');

% 输出分析结果
fprintf('\n%s与%s的相关性分析结果：\n', var1_name, var2_name);
fprintf('1. 协方差: %.4f (Matlab验证: %.4f)\n', covariance, matlab_cov);
fprintf('2. Pearson相关系数: %.4f (Matlab验证: %.4f)，P值: %.8f\n', ...
        pearson_corr, pearson_corr_matlab, pearson_p);
fprintf('3. Spearman相关系数: %.4f (Matlab验证: %.4f)，P值: %.8f\n', ...
        spearman_corr, spearman_corr_matlab, spearman_p);

% 结果解释
alpha = 0.05;
if pearson_p < alpha
    pearson_conclusion = sprintf('显著的线性相关（P=%.8f < %.2f）', pearson_p, alpha);
else
    pearson_conclusion = sprintf('无显著的线性相关（P=%.8f ≥ %.2f）', pearson_p, alpha);
end

if spearman_p < alpha
    spearman_conclusion = sprintf('显著的单调相关（P=%.8f < %.2f）', spearman_p, alpha);
else
    spearman_conclusion = sprintf('无显著的单调相关（P=%.8f ≥ %.2f）', spearman_p, alpha);
end

fprintf('\n结果解释：\n');
fprintf('- Pearson相关系数表明%s与%s存在%s\n', var1_name, var2_name, pearson_conclusion);
fprintf('- Spearman相关系数表明%s与%s存在%s\n', var1_name, var2_name, spearman_conclusion);

% 可视化相关性
figure('Position', [100 100 1000 800]);

% 1. 散点图与回归线
subplot(2, 2, 1);
scatter(x, y, 'b', 'MarkerFaceColor', 'b', 'AlphaData', 0.6);
hold on;
% 添加回归线
p = polyfit(x, y, 1);
y_fit = polyval(p, x);
plot(x, y_fit, 'r', 'LineWidth', 2);
title(sprintf('%s与%s的散点图及回归线', var1_name, var2_name));
xlabel(var1_name);
ylabel(var2_name);
grid on;
hold off;

% 2. 相关性热图
subplot(2, 2, 2);
corr_matrix = [1 pearson_corr; pearson_corr 1];
heatmap(corr_matrix, ...
        'RowLabels', {var1_name, var2_name}, ...
        'ColumnLabels', {var1_name, var2_name}, ...
        'ColorbarVisible', 'off');
title('相关性热图');

% 3. 秩次散点图（用于Spearman）
subplot(2, 2, 3);
scatter(x_rank, y_rank, 'g', 'MarkerFaceColor', 'g', 'AlphaData', 0.6);
hold on;
% 添加秩次回归线
p_rank = polyfit(x_rank, y_rank, 1);
y_rank_fit = polyval(p_rank, x_rank);
plot(x_rank, y_rank_fit, 'darkgreen', 'LineWidth', 2);
title(sprintf('%s与%s的秩次散点图', var1_name, var2_name));
xlabel(sprintf('%s的秩次', var1_name));
ylabel(sprintf('%s的秩次', var2_name));
grid on;
hold off;

% 4. 相关性数值对比
subplot(2, 2, 4);
methods = {'协方差', 'Pearson相关系数', 'Spearman相关系数'};
values = [covariance, pearson_corr, spearman_corr];
bar(values, 'FaceColor', 'flat', 'CData', [0.2 0.4 0.6; 0.8 0.4 0; 0 0.6 0.2]);
set(gca, 'XTickLabel', methods, 'XTick', 1:3);
title('三种相关性度量值对比');
grid on;
ylim([min(values)*1.1, max(values)*1.1]);

tight_layout;
    