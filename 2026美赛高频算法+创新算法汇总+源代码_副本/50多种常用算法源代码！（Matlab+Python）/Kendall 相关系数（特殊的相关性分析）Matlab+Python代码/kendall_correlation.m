% Kendall相关系数分析案例说明：
% 本案例实现Kendall秩相关系数（τ系数）分析，这是一种非参数统计方法，用于衡量
% 两个变量之间的单调相关程度。与Spearman相关系数相比，Kendall更适用于样本量
% 较小或存在较多相同秩次（平局）的数据。
%
% Kendall系数取值范围为[-1, 1]：
% - 1表示完全正相关
% - -1表示完全负相关
% - 0表示无单调相关
%
% 示例使用汽车数据集，分析汽车重量与油耗之间的相关性，并与Pearson和Spearman
% 系数进行对比，展示不同相关系数的特点。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行完整分析');
end

% 加载数据集
disp('加载汽车数据集...');
try
    % 尝试加载内置数据集
    load mpg.mat;
    x = mpg.Weight;       % 汽车重量
    y = mpg.MPG;          % 每加仑英里数
    var1_name = '汽车重量';
    var2_name = '每加仑英里数(mpg)';
catch
    % 生成模拟数据（汽车重量与油耗负相关）
    rng(42);  % 设置随机种子
    n = 300;
    x = 2000 + 3000 * rand(n, 1);  % 汽车重量：2000-5000磅
    y = 40 - 0.005 * x + 3 * randn(n, 1);  % 每加仑英里数（与重量负相关）
    var1_name = '汽车重量';
    var2_name = '每加仑英里数(mpg)';
end

% 数据预处理：移除缺失值
valid_idx = ~isnan(x) & ~isnan(y);
x = x(valid_idx);
y = y(valid_idx);

% 手动计算Kendall相关系数（简化版，处理小样本）
function tau = kendall_correlation(x, y)
    n = length(x);
    concordant = 0;  % 一致对数量
    discordant = 0;  % 不一致对数量
    
    % 遍历所有数据对
    for i = 1:n
        for j = i+1:n
            % 计算x和y的差异符号
            dx = x(i) - x(j);
            dy = y(i) - y(j);
            
            if dx * dy > 0
                % 符号相同，一致对
                concordant = concordant + 1;
            elseif dx * dy < 0
                % 符号相反，不一致对
                discordant = discordant + 1;
            end
            % 若dx或dy为0，视为平局，不纳入计算
        end
    end
    
    % Kendall系数公式：τ = (一致对 - 不一致对) / C(n,2)
    total_pairs = n * (n - 1) / 2;
    if total_pairs == 0
        tau = 0;
    else
        tau = (concordant - discordant) / total_pairs;
    end
end

% 计算三种相关系数
kendall_manual = kendall_correlation(x(1:50), y(1:50));  % 手动计算（小样本）
[kendall_tau, kendall_p] = corr(x, y, 'Type', 'Kendall');  % Matlab函数计算
[pearson_r, pearson_p] = corr(x, y, 'Type', 'Pearson');
[spearman_rho, spearman_p] = corr(x, y, 'Type', 'Spearman');

% 输出分析结果
fprintf('\n%s与%s的相关性分析结果：\n', var1_name, var2_name);
fprintf('Kendall相关系数（手动计算，小样本）: %.4f\n', kendall_manual);
fprintf('Kendall相关系数（τ）: %.4f，P值: %.8f\n', kendall_tau, kendall_p);
fprintf('Pearson相关系数（r）: %.4f，P值: %.8f\n', pearson_r, pearson_p);
fprintf('Spearman相关系数（ρ）: %.4f，P值: %.8f\n', spearman_rho, spearman_p);

% 结果解释
alpha = 0.05;
if kendall_p < alpha
    kendall_conclusion = sprintf('显著的Kendall单调相关（P=%.8f < %.2f）', kendall_p, alpha);
else
    kendall_conclusion = sprintf('无显著的Kendall单调相关（P=%.8f ≥ %.2f）', kendall_p, alpha);
end
fprintf('\n结果解释：%s与%s存在%s\n', var1_name, var2_name, kendall_conclusion);

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
title(sprintf('%s与%s的散点图及回归线', var1_name, var2_name), 'FontSize', 12);
xlabel(var1_name);
ylabel(var2_name);
grid on;
hold off;

% 2. 三种相关系数对比
subplot(2, 2, 2);
methods = {'Kendall τ', 'Pearson r', 'Spearman ρ'};
values = [kendall_tau, pearson_r, spearman_rho];
bar(values, 'FaceColor', 'flat', 'CData', [0.2 0.4 0.6; 0.8 0.4 0; 0 0.6 0.2]);
set(gca, 'XTickLabel', methods, 'XTick', 1:3, 'YLim', [-1.1 1.1]);
title('三种相关系数对比', 'FontSize', 12);
grid on;
yline(0, 'k--', 'LineWidth', 1);

% 3. 秩次散点图（用于Kendall和Spearman）
subplot(2, 2, 3);
x_rank = tiedrank(x);
y_rank = tiedrank(y);
scatter(x_rank, y_rank, 'm', 'MarkerFaceColor', 'm', 'AlphaData', 0.6);
hold on;
% 添加秩次回归线
p_rank = polyfit(x_rank, y_rank, 1);
y_rank_fit = polyval(p_rank, x_rank);
plot(x_rank, y_rank_fit, 'darkred', 'LineWidth', 2);
title(sprintf('%s与%s的秩次散点图', var1_name, var2_name), 'FontSize', 12);
xlabel(sprintf('%s的秩次', var1_name));
ylabel(sprintf('%s的秩次', var2_name));
grid on;
hold off;

% 4. 相关性热图
subplot(2, 2, 4);
corr_matrix = [1 kendall_tau; kendall_tau 1];
heatmap(corr_matrix, ...
        'RowLabels', {var1_name, var2_name}, ...
        'ColumnLabels', {var1_name, var2_name}, ...
        'ColorbarVisible', 'off');
title('Kendall相关系数热图', 'FontSize', 12);

tight_layout;
    