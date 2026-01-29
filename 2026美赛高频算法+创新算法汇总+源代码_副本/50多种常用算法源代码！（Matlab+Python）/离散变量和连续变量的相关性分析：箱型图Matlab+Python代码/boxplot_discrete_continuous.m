% 箱型图分析离散与连续变量相关性案例说明：
% 本案例使用箱型图分析离散变量与连续变量之间的关系。箱型图通过展示连续变量在
% 不同离散类别中的分布差异（如中位数、四分位数、异常值等）来判断两者是否相关。
% 分布差异越明显，表明两个变量的相关性越强。
%
% 示例使用泰坦尼克号数据集，分析"乘客等级"（离散变量）与"年龄"（连续变量）之间
% 的关系，通过比较不同乘客等级的年龄分布来判断它们是否相关。

% 检查统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行完整分析');
end

% 加载泰坦尼克号数据集（如果没有内置数据集，使用模拟数据）
disp('加载泰坦尼克号数据集...');
try
    % 尝试加载内置数据集
    load titanic.mat;
    % 提取需要的变量：pclass（乘客等级）和age（年龄）
    pclass = titanic.Pclass;
    age = titanic.Age;
catch
    % 生成模拟数据
    rng(42);  % 设置随机种子
    n = 500;
    pclass = randi([1,3], n, 1);  % 乘客等级：1,2,3
    % 不同等级的年龄分布有差异
    age = zeros(n, 1);
    for i = 1:n
        if pclass(i) == 1
            age(i) = normrnd(38, 12);  % 1等舱乘客年龄较大
        elseif pclass(i) == 2
            age(i) = normrnd(30, 10);  % 2等舱乘客年龄中等
        else
            age(i) = normrnd(25, 8);   % 3等舱乘客年龄较小
        end
    end
    % 添加一些缺失值
    age(randi([1,n], 50, 1)) = NaN;
end

% 数据预处理：移除缺失值
valid_idx = ~isnan(age) & ~isnan(pclass);
pclass = pclass(valid_idx);
age = age(valid_idx);
pclass = categorical(pclass);  % 转换为分类变量

% 按离散变量分组，获取连续变量数据
classes = unique(pclass);
groups = cell(length(classes), 1);
for i = 1:length(classes)
    groups{i} = age(pclass == classes(i));
end

% 计算各组基本统计量
stats_names = {'样本量', '均值', '中位数', '标准差', '最小值', '25%分位数', '75%分位数', '最大值'};
stats_summary = zeros(length(stats_names), length(classes));

for i = 1:length(classes)
    data = groups{i};
    stats_summary(1, i) = length(data);                % 样本量
    stats_summary(2, i) = mean(data);                  % 均值
    stats_summary(3, i) = median(data);                % 中位数
    stats_summary(4, i) = std(data);                   % 标准差
    stats_summary(5, i) = min(data);                   % 最小值
    stats_summary(6, i) = prctile(data, 25);           % 25%分位数
    stats_summary(7, i) = prctile(data, 75);           % 75%分位数
    stats_summary(8, i) = max(data);                   % 最大值
end

% 显示统计量
fprintf('\n不同乘客等级的年龄统计量：\n');
disp(array2table(round(stats_summary, 2), ...
    'RowNames', stats_names, ...
    'VariableNames', cellfun(@(x) sprintf('等级%d', x), num2cell(classes), 'UniformOutput', false)));

% 执行单因素方差分析（ANOVA），检验组间差异是否显著
[p_val, f_val] = anova1(cat(2, groups{:}));
fprintf('\nANOVA检验结果：F值 = %.4f, P值 = %.8f\n', f_val, p_val);

% 结果解释
alpha = 0.05;
if p_val < alpha
    conclusion = sprintf('由于P值(%.8f) < %.2f，不同乘客等级的年龄分布存在显著差异，表明两者相关。', p_val, alpha);
else
    conclusion = sprintf('由于P值(%.8f) ≥ %.2f，不同乘客等级的年龄分布无显著差异，表明两者不相关。', p_val, alpha);
end
fprintf('结论：%s\n', conclusion);

% 绘制箱型图可视化
figure('Position', [100 100 800 600]);
boxplot(cat(2, groups{:}), 'Labels', cellfun(@(x) sprintf('等级%d', x), num2cell(classes), 'UniformOutput', false), ...
        'Colors', [0.2 0.4 0.6], 'Symbol', 'o', 'MarkerSize', 5, ...
        'MeanMarker', 'o', 'MeanMarkerSize', 6, 'MeanEdgeColor', 'r');

title('不同乘客等级的年龄分布箱型图', 'FontSize', 14);
xlabel('乘客等级', 'FontSize', 12);
ylabel('年龄', 'FontSize', 12);
grid on;
annotation('textbox', [0.35, 0.9, 0.3, 0.05], ...
    'String', sprintf('ANOVA: F=%.2f, P=%.6f', f_val, p_val), ...
    'EdgeColor', 'none', 'BackgroundColor', 'w', 'HorizontalAlignment', 'center');

% 补充：绘制小提琴图（展示更详细的分布形状）
figure('Position', [100 100 800 600]);
violinplot(cat(2, groups{:}));
set(gca, 'XTick', 1:length(classes), ...
         'XTickLabel', cellfun(@(x) sprintf('等级%d', x), num2cell(classes), 'UniformOutput', false));
title('不同乘客等级的年龄分布小提琴图', 'FontSize', 14);
xlabel('乘客等级', 'FontSize', 12);
ylabel('年龄', 'FontSize', 12);
grid on;
    