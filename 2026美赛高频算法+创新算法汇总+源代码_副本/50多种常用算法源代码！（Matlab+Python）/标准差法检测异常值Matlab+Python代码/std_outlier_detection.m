% 标准差法检测异常值案例说明：
% 本案例使用标准差法检测数据中的异常值。标准差法假设数据服从正态分布，
% 通常将偏离均值超过3个标准差的数据点视为异常值（3σ原则）。
% 示例中生成服从正态分布的模拟数据并添加异常值，使用3σ原则进行检测。

% 设置随机种子，确保结果可复现
rng(42);

% 生成示例数据：正态分布数据 + 异常值
mu = 50;                  % 均值
sigma = 10;               % 标准差
normal_data = normrnd(mu, sigma, 300, 1);  % 正常数据
outliers = [10; 20; 90; 100; 110; 120];    % 异常值
data = [normal_data; outliers];            % 合并数据

% 计算均值和标准差
mean_val = mean(data);
std_val = std(data);

% 定义异常值阈值（3σ原则）
lower_threshold = mean_val - 3 * std_val;
upper_threshold = mean_val + 3 * std_val;

% 检测异常值
outlier_indices = find((data < lower_threshold) | (data > upper_threshold));
outlier_values = data(outlier_indices);
normal_values = data(find((data >= lower_threshold) & (data <= upper_threshold)));

% 输出结果
fprintf('标准差法(3σ原则)异常值检测结果：\n');
fprintf('数据总量: %d 个\n', length(data));
fprintf('均值(μ): %.2f\n', mean_val);
fprintf('标准差(σ): %.2f\n', std_val);
fprintf('异常值下限: μ - 3σ = %.2f\n', lower_threshold);
fprintf('异常值上限: μ + 3σ = %.2f\n', upper_threshold);
fprintf('检测到异常值数量: %d 个\n', length(outlier_values));
fprintf('异常值: ');
for i = 1:length(outlier_values)
    fprintf('%.2f ', outlier_values(i));
end
fprintf('\n');

% 可视化结果
figure;
edges = linspace(min(data), max(data), 31);  % 直方图区间
[N, bins] = histcounts(data, edges);

% 绘制直方图并区分正常数据和异常值
for i = 1:length(N)
    bin_center = (bins(i) + bins(i+1)) / 2;
    if bin_center < lower_threshold || bin_center > upper_threshold
        bar(bin_center, N(i), (bins(i+1)-bins(i))*0.9, 'FaceColor', 'r', 'EdgeColor', 'k');
    else
        bar(bin_center, N(i), (bins(i+1)-bins(i))*0.9, 'FaceColor', 'b', 'EdgeColor', 'k');
    end
    hold on;
end

% 绘制均值和3σ线
line([mean_val mean_val], [0 max(N)], 'Color', 'g', 'LineWidth', 2);
line([lower_threshold lower_threshold], [0 max(N)], 'Color', 'orange', 'LineStyle', '--', 'LineWidth', 2);
line([upper_threshold upper_threshold], [0 max(N)], 'Color', 'orange', 'LineStyle', '--', 'LineWidth', 2);

% 添加标签和图例
title('标准差法(3σ原则)异常值检测');
xlabel('数据值');
ylabel('频数');
legend('正常数据', '异常值', sprintf('均值 (μ = %.2f)', mean_val), ...
       sprintf('3σ 边界'), 'Location', 'best');
grid on;
hold off;
    