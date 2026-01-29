% 箱型图检测异常值案例说明：
% 本案例使用箱型图方法检测数据中的异常值。箱型图基于数据的四分位数，
% 通过计算上下限（Q1-1.5*IQR和Q3+1.5*IQR）来识别异常值，其中IQR是四分位距。
% 示例中生成包含异常值的随机数据，使用箱型图进行可视化并标记异常值。

% 设置随机种子，确保结果可复现
rng(42);

% 生成示例数据：主要分布在[10, 50]之间，添加一些异常值
normal_data = normrnd(30, 8, 200, 1);  % 正常数据
outliers = [5; 60; 65; 70; -2; 75];    % 异常值
data = [normal_data; outliers];        % 合并数据

% 计算四分位数和异常值边界
q1 = prctile(data, 25);  % 第一四分位数
q3 = prctile(data, 75);  % 第三四分位数
iqr = q3 - q1;           % 四分位距
lower_bound = q1 - 1.5 * iqr;  % 下限
upper_bound = q3 + 1.5 * iqr;  % 上限

% 检测异常值
outlier_indices = find((data < lower_bound) | (data > upper_bound));
outlier_values = data(outlier_indices);

% 输出结果
fprintf('箱型图异常值检测结果：\n');
fprintf('数据总量: %d 个\n', length(data));
fprintf('第一四分位数(Q1): %.2f\n', q1);
fprintf('第三四分位数(Q3): %.2f\n', q3);
fprintf('四分位距(IQR): %.2f\n', iqr);
fprintf('异常值下限: %.2f\n', lower_bound);
fprintf('异常值上限: %.2f\n', upper_bound);
fprintf('检测到异常值数量: %d 个\n', length(outlier_values));
fprintf('异常值: ');
for i = 1:length(outlier_values)
    fprintf('%.2f ', outlier_values(i));
end
fprintf('\n');

% 可视化箱型图
figure;
boxplot(data, 'Colors', [0 0 1], 'Marker', 'o', 'MarkerSize', 8, ...
        'Symbol', 'r+', 'Whisker', 1.5);
title('箱型图异常值检测');
ylabel('数据值');
grid on;

% 添加文本说明
annotation('textbox', [0.15, 0.85, 0.15, 0.05], 'String', sprintf('Q1: %.2f', q1), ...
           'EdgeColor', 'none', 'BackgroundColor', 'w');
annotation('textbox', [0.15, 0.1, 0.15, 0.05], 'String', sprintf('下限: %.2f', lower_bound), ...
           'EdgeColor', 'none', 'BackgroundColor', 'w');
annotation('textbox', [0.15, 0.9, 0.15, 0.05], 'String', sprintf('上限: %.2f', upper_bound), ...
           'EdgeColor', 'none', 'BackgroundColor', 'w');
annotation('textbox', [0.15, 0.75, 0.15, 0.05], 'String', sprintf('Q3: %.2f', q3), ...
           'EdgeColor', 'none', 'BackgroundColor', 'w');
    