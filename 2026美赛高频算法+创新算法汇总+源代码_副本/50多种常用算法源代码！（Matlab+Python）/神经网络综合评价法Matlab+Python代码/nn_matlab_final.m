% 神经网络评价算法案例说明：
% 本案例使用神经网络对10个城市的发展水平进行评价建模，
% 基于5项指标（GDP、人口、就业率、教育投入、医疗资源）训练模型，
% 然后用训练好的模型对5个新城市进行发展水平评分和排序。
% 神经网络通过学习已知评分的城市数据，能够捕捉指标间的非线性关系，
% 提供更精准的综合评价结果。

% 设置随机种子，确保结果可复现
rng(42);

% 1. 准备训练数据
% 训练数据：10个城市，5项指标（GDP(亿元)、人口(万)、就业率(%)、教育投入(亿元)、医疗资源(床位数/万人)
train_data = [
    3200, 280, 92.5, 85, 6.2;
    2800, 220, 91.3, 78, 5.8;
    4100, 350, 93.1, 92, 6.5;
    1800, 150, 89.7, 65, 4.9;
    5200, 420, 94.2, 105, 7.1;
    2500, 200, 90.5, 72, 5.5;
    3800, 320, 92.8, 88, 6.3;
    1500, 130, 88.9, 60, 4.7;
    4500, 380, 93.5, 96, 6.8;
    2200, 180, 89.9, 68, 5.2];

% 训练标签：专家对10个城市的发展水平评分（0-100分）
train_labels = [78, 72, 85, 65, 92, 69, 82, 62, 88, 67];

% 2. 准备测试数据（5个新城市）
test_data = [
    3600, 300, 92.2, 86, 6.4;   % 城市A
    2900, 230, 91.5, 79, 5.9;   % 城市B
    4800, 400, 93.8, 98, 6.9;   % 城市C
    2100, 170, 89.8, 67, 5.1;   % 城市D
    3300, 290, 92.0, 84, 6.3];  % 城市E

% 3. 数据标准化（均值为0，标准差为1）
[train_data_scaled, mu, sigma] = zscore(train_data);  % 标准化训练数据
% 标准化测试数据（使用训练数据的均值和标准差）
test_data_scaled = (test_data - repmat(mu, size(test_data, 1), 1)) ...
    ./ repmat(sigma, size(test_data, 1), 1);

% 4. 创建并训练神经网络模型
% 创建前馈神经网络，1个隐藏层含10个神经元
model = feedforwardnet(10);
model.layers{1}.transferFcn = 'relu';  % 隐藏层激活函数
model.trainFcn = 'trainadam';          % 优化器
model.maxEpochs = 1000;                % 最大迭代次数
model.trainParam.lr = 0.001;           % 学习率
model.performFcn = 'mse';              % 性能函数

% 训练模型
model = train(model, train_data_scaled', train_labels');

% 5. 预测测试数据
test_scores = model(test_data_scaled')';

% 6. 计算模型在训练数据上的拟合优度
train_pred = model(train_data_scaled')';
ss_total = sum((train_labels - mean(train_labels)).^2);
ss_residual = sum((train_labels - train_pred).^2);
r_squared = 1 - ss_residual/ss_total;

% 7. 排序（从高到低）
[sorted_scores, sorted_indices] = sort(test_scores, 'descend');

% 输出结果
disp("神经网络评价算法结果：");
fprintf("模型在训练数据上的R²得分: %.4f\n", r_squared);
disp("\n测试城市的发展水平评分及排名：");
city_names = ['A'; 'B'; 'C'; 'D'; 'E'];  % 城市名称
for i = 1:length(sorted_indices)
    fprintf("第%d名: 城市%c, 评分: %.2f\n", ...
        i, city_names(sorted_indices(i)), sorted_scores(i));
end
