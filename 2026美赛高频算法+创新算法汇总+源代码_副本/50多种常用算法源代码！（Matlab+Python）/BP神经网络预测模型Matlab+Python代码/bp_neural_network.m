% BP神经网络预测模型案例说明：
% 本案例使用BP神经网络预测某商品的销售量，基于过去12个月的
% 广告投入、促销活动强度和节假日因素等特征数据。BP神经网络
% 是一种多层前馈神经网络，通过反向传播算法调整权重，适用于
% 复杂非线性关系的预测问题。本案例使用1个隐藏层(10个神经元)，
% 输入层3个神经元(对应3个特征)，输出层1个神经元(预测销售量)。

% 1. 准备数据
% 设置随机种子，确保结果可复现
rng(42);

% 特征数据：[广告投入(万元), 促销强度(0-10), 节假日数量]
X = [
    5.2, 7, 2; 6.3, 5, 1; 4.8, 8, 3; 7.1, 6, 2;
    5.9, 9, 1; 8.2, 7, 2; 6.8, 6, 3; 9.1, 8, 2;
    7.5, 10, 1; 8.9, 7, 2; 9.5, 9, 3; 10.2, 8, 2
];

% 目标数据：销售量(千件)，与特征呈非线性关系
y = [
    12.5; 11.8; 13.2; 14.5; 
    13.8; 16.2; 15.1; 17.5; 
    16.8; 18.2; 19.5; 20.3
];

% 2. 数据预处理：归一化到[0,1]范围
[X_scaled, X_means, X_stds] = normalize_data(X);
[y_scaled, y_mean, y_std] = normalize_data(y);

% 3. 划分训练集和测试集
indices = randperm(size(X, 1));
train_ratio = 0.8;
train_size = round(train_ratio * size(X, 1));
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

X_train = X_scaled(train_indices, :);
y_train = y_scaled(train_indices, :);
X_test = X_scaled(test_indices, :);
y_test = y_scaled(test_indices, :);

% 4. 创建并训练BP神经网络模型
% 定义模型：1个隐藏层(10个神经元)，使用ReLU激活函数
hidden_layer_size = 10;
net = feedforwardnet(hidden_layer_size);
net.layers{1}.transferFcn = 'relu';  % 隐藏层激活函数
net.trainFcn = 'trainadam';          % 训练算法
net.performFcn = 'mse';              % 性能函数
net.trainParam.epochs = 1000;        % 最大迭代次数
net.trainParam.showWindow = false;   % 不显示训练窗口

% 训练模型
net = train(net, X_train', y_train');

% 5. 模型预测
% 对训练集和测试集进行预测
y_train_pred_scaled = net(X_train')';
y_test_pred_scaled = net(X_test')';

% 反归一化预测结果
y_train_pred = denormalize_data(y_train_pred_scaled, y_mean, y_std);
y_test_pred = denormalize_data(y_test_pred_scaled, y_mean, y_std);
y_train_actual = denormalize_data(y_train, y_mean, y_std);
y_test_actual = denormalize_data(y_test, y_mean, y_std);

% 6. 模型评估
% 计算训练集和测试集的性能指标
train_rmse = sqrt(mean((y_train_actual - y_train_pred).^2));
test_rmse = sqrt(mean((y_test_actual - y_test_pred).^2));
train_r2 = 1 - sum((y_train_actual - y_train_pred).^2) / sum((y_train_actual - mean(y_train_actual)).^2);
test_r2 = 1 - sum((y_test_actual - y_test_pred).^2) / sum((y_test_actual - mean(y_test_actual)).^2);

% 7. 预测未来数据
% 未来3个月的特征数据
future_X = [
    11.0, 9, 2; 11.5, 8, 1; 12.0, 10, 3
];
% 归一化未来特征数据
future_X_scaled = (future_X - X_means) ./ X_stds;
% 预测并反归一化
future_y_scaled = net(future_X_scaled')';
future_y = denormalize_data(future_y_scaled, y_mean, y_std);

% 8. 输出结果
disp("BP神经网络预测模型结果：");
fprintf("模型结构：输入层%d个神经元，隐藏层%d个神经元，输出层1个神经元\n", ...
    size(X, 2), hidden_layer_size);
fprintf("训练集RMSE: %.4f, R²: %.4f\n", train_rmse, train_r2);
fprintf("测试集RMSE: %.4f, R²: %.4f\n", test_rmse, test_r2);

disp("\n训练集预测结果：");
for i = 1:size(y_train_actual, 1)
    fprintf("实际值: %.2f, 预测值: %.2f, 误差: %.2f\n", ...
        y_train_actual(i), y_train_pred(i), y_train_actual(i)-y_train_pred(i));
end

disp("\n测试集预测结果：");
for i = 1:size(y_test_actual, 1)
    fprintf("实际值: %.2f, 预测值: %.2f, 误差: %.2f\n", ...
        y_test_actual(i), y_test_pred(i), y_test_actual(i)-y_test_pred(i));
end

disp("\n未来3个月销售量预测结果：");
for i = 1:size(future_y, 1)
    fprintf("第%d个月: 广告投入%.1f万元, 促销强度%d, 节假日%d天, 预测销售量%.2f千件\n", ...
        i, future_X(i,1), future_X(i,2), future_X(i,3), future_y(i));
end

% 9. 可视化结果
figure;
% 绘制训练集结果
plot(1:size(y_train_actual, 1), y_train_actual, 'bo-', 'MarkerSize', 6, ...
    'LineWidth', 1.5, 'DisplayName', '训练集实际值');
hold on;
plot(1:size(y_train_pred, 1), y_train_pred, 'r--', 'LineWidth', 1.5, ...
    'DisplayName', '训练集预测值');
% 绘制测试集结果（偏移显示）
test_offset = size(y_train_actual, 1);
plot(test_offset+1:test_offset+size(y_test_actual, 1), y_test_actual, 'go-', ...
    'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '测试集实际值');
plot(test_offset+1:test_offset+size(y_test_pred, 1), y_test_pred, 'm--', ...
    'LineWidth', 1.5, 'DisplayName', '测试集预测值');
% 绘制未来预测结果（偏移显示）
future_offset = test_offset + size(y_test_actual, 1);
plot(future_offset+1:future_offset+size(future_y, 1), future_y, 'c*-', ...
    'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', '未来预测值');

xlabel('样本索引');
ylabel('销售量（千件）');
title('BP神经网络销售量预测');
legend();
grid on;
hold off;

% 归一化函数
function [normalized, means, stds] = normalize_data(data)
    means = mean(data);
    stds = std(data);
    % 避免除以零
    stds(stds < 1e-10) = 1;
    normalized = (data - means) ./ stds;
end

% 反归一化函数
function denormalized = denormalize_data(normalized, mean_val, std_val)
    % 避免乘以零
    if std_val < 1e-10
        std_val = 1;
    end
    denormalized = normalized * std_val + mean_val;
end
