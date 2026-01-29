% SOM自组织神经网络案例说明：
% 自组织映射（SOM）是一种无监督的人工神经网络，通过竞争学习将高维数据映射到低维（通常是2D）空间，
% 同时保留数据的拓扑结构。SOM由网格状排列的神经元组成，每个神经元有一个与输入数据维度相同的权重向量。
%
% 算法步骤：
% 1. 初始化神经元权重向量
% 2. 从数据集中随机选择一个样本
% 3. 找到与样本最相似的神经元（最佳匹配单元，BMU）
% 4. 更新BMU及其邻域神经元的权重，使其更接近样本
% 5. 逐渐减小邻域大小和学习率
% 6. 重复步骤2-5直到收敛
%
% 本案例使用鸢尾花数据集（4维特征），通过SOM映射到10×10的二维网格，
% 可视化聚类结果和数据的拓扑结构。

% 检查神经网络工具箱
if ~license('test', 'Neural_Network_Toolbox')
    error('需要Neural Network Toolbox才能运行SOM');
end

% 加载数据集
load fisheriris;
X = meas;  % 特征数据
y_true = species;  % 真实标签（仅用于对比）
feature_names = {'花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'};

% 数据归一化（SOM对数据尺度敏感，通常归一化到[0,1]）
X_scaled = mapminmax(X', 0, 1)';  % 归一化到[0,1]

% SOM参数设置
grid_size = [10 10];  % 神经元网格大小 [行, 列]
input_dim = size(X_scaled, 2);  % 输入维度（4）
num_iterations = 1000;  % 迭代次数

% 创建并训练SOM网络
disp('训练SOM神经网络...');
net = selforgmap(grid_size);  % 创建SOM网络
net.trainParam.epochs = num_iterations;  % 设置迭代次数
net = train(net, X_scaled');  % 训练网络（注意输入为行向量）

% 对数据进行映射，获取每个样本的最佳匹配单元（BMU）
[bmus, ~] = net(X_scaled');  % BMU的索引，大小为[2, 样本数]
% 转换为坐标（Matlab的SOM索引从1开始）
bmu_coords = zeros(size(X_scaled, 1), 2);
for i = 1:size(X_scaled, 1)
    [r, c] = ind2sub(grid_size, bmus(i));
    bmu_coords(i, :) = [r, c];
end

% 可视化SOM聚类结果
figure('Position', [100 100 1000 800]);

% 1. SOM节点激活频率热图
activation_map = zeros(grid_size);
for i = 1:size(X_scaled, 1)
    r = bmu_coords(i, 1);
    c = bmu_coords(i, 2);
    activation_map(r, c) = activation_map(r, c) + 1;
end

subplot(2, 2, 1);
imagesc(activation_map);
colormap(viridis);
colorbar;
title('SOM节点激活频率');
xlabel('神经元列索引');
ylabel('神经元行索引');
axis xy;  % 使y轴从上到下递增

% 2. 样本在SOM上的分布（按真实标签）
subplot(2, 2, 2);
colors = lines(3);
unique_species = unique(y_true);
for i = 1:length(unique_species)
    idx = strcmp(y_true, unique_species{i});
    plot(bmu_coords(idx, 2), bmu_coords(idx, 1), 'o', ...
        'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'DisplayName', unique_species{i});
    hold on;
end
title('样本在SOM上的分布（按真实标签）');
xlabel('神经元列索引');
ylabel('神经元行索引');
xlim([0.5 grid_size(2)+0.5]);
ylim([0.5 grid_size(1)+0.5]);
xticks(1:grid_size(2));
yticks(1:grid_size(1));
grid on;
legend('Location', 'best');
hold off;
axis xy;

% 3. SOM的U矩阵（反映节点间相似度）
% 计算U矩阵：每个节点与周围节点的平均距离
u_matrix = zeros(grid_size);
for r = 1:grid_size(1)
    for c = 1:grid_size(2)
        % 获取当前节点的权重
        idx = sub2ind(grid_size, r, c);
        w = net.IW{1,1}(:, idx);
        
        % 计算与相邻节点的距离
        neighbors = [];
        if r > 1
            neighbors = [neighbors; sub2ind(grid_size, r-1, c)];
        end
        if r < grid_size(1)
            neighbors = [neighbors; sub2ind(grid_size, r+1, c)];
        end
        if c > 1
            neighbors = [neighbors; sub2ind(grid_size, r, c-1)];
        end
        if c < grid_size(2)
            neighbors = [neighbors; sub2ind(grid_size, r, c+1)];
        end
        
        % 计算平均距离
        dists = zeros(length(neighbors), 1);
        for k = 1:length(neighbors)
            w_neigh = net.IW{1,1}(:, neighbors(k));
            dists(k) = norm(w - w_neigh);
        end
        u_matrix(r, c) = mean(dists);
    end
end

subplot(2, 2, 3);
imagesc(u_matrix);
colormap(bone);
colorbar;
title('SOM的U矩阵（反映拓扑结构）');
xlabel('神经元列索引');
ylabel('神经元行索引');
axis xy;

% 4. 第一个特征的权重分布
feature_idx = 1;
weights_map = zeros(grid_size);
for r = 1:grid_size(1)
    for c = 1:grid_size(2)
        idx = sub2ind(grid_size, r, c);
        weights_map(r, c) = net.IW{1,1}(feature_idx, idx);
    end
end

subplot(2, 2, 4);
imagesc(weights_map);
colormap(coolwarm);
colorbar;
title(sprintf('特征 "%s" 的权重分布', feature_names{feature_idx}));
xlabel('神经元列索引');
ylabel('神经元行索引');
axis xy;

% 展示所有特征的权重分布
figure('Position', [100 200 1200 800]);
for i = 1:input_dim
    subplot(2, 2, i);
    weights_map = zeros(grid_size);
    for r = 1:grid_size(1)
        for c = 1:grid_size(2)
            idx = sub2ind(grid_size, r, c);
            weights_map(r, c) = net.IW{1,1}(i, idx);
        end
    end
    imagesc(weights_map);
    colormap(coolwarm);
    colorbar;
    title(sprintf('特征 "%s" 的权重分布', feature_names{i}));
    xlabel('神经元列索引');
    ylabel('神经元行索引');
    axis xy;
end
    