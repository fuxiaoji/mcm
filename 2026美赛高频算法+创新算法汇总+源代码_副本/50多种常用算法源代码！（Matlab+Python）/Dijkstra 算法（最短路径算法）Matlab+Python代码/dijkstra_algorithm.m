% Dijkstra算法案例说明：
% Dijkstra算法是一种用于寻找图中从单个源节点到其他所有节点的最短路径的贪心算法。
% 它适用于边权值为非负数的图，通过不断选择当前距离源节点最近的未访问节点，
% 并更新其相邻节点的距离来实现。
%
% 算法步骤：
% 1. 初始化：将源节点到所有其他节点的距离设为无穷大，源节点到自身的距离设为0
% 2. 选择距离源节点最近的未访问节点u
% 3. 对于u的每个相邻节点v，计算从源节点经过u到v的距离，若小于当前已知距离则更新
% 4. 标记u为已访问
% 5. 重复步骤2-4，直到所有节点都被访问
%
% 本案例使用一个包含7个节点的无向图，演示Dijkstra算法如何找到从源节点到其他所有节点的最短路径，
% 并可视化图结构和最短路径结果。

% 定义图的邻接矩阵（表示节点间的距离，Inf表示无直接连接）
% 节点1-7代表7个不同的地点
graph = [
    0   4   Inf Inf Inf Inf 2;   % 节点1与其他节点的距离
    4   0   6   Inf Inf Inf Inf; % 节点2与其他节点的距离
    Inf 6   0   3   Inf 5   Inf; % 节点3与其他节点的距离
    Inf Inf 3   0   2   Inf Inf; % 节点4与其他节点的距离
    Inf Inf Inf 2   0   1   Inf; % 节点5与其他节点的距离
    Inf Inf 5   Inf 1   0   4;   % 节点6与其他节点的距离
    2   Inf Inf Inf Inf 4   0    % 节点7与其他节点的距离
];

% 节点名称（用于可视化）
node_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
num_nodes = size(graph, 1);

% 选择起点（例如选择节点1，对应名称'A'）
start_node = 1;
fprintf('使用Dijkstra算法计算从节点%s到其他所有节点的最短路径...\n', node_names{start_node});

% 初始化距离数组，Inf表示不可达
distances = Inf(1, num_nodes);
% 起点到自身的距离为0
distances(start_node) = 0;
% 记录已访问的节点
visited = false(1, num_nodes);
% 记录最短路径的前驱节点
predecessors = zeros(1, num_nodes);
predecessors(:) = NaN;

for i = 1:num_nodes
    % 找到当前未访问节点中距离最小的节点u
    min_distance = Inf;
    u = -1;
    for j = 1:num_nodes
        if ~visited(j) && distances(j) < min_distance
            min_distance = distances(j);
            u = j;
        end
    end
    
    % 如果所有节点都已访问或剩下的节点不可达，退出循环
    if u == -1 || min_distance == Inf
        break;
    end
    
    % 标记u为已访问
    visited(u) = true;
    
    % 更新u的所有邻居节点的距离
    for v = 1:num_nodes
        % 如果v是u的邻居且未被访问
        if graph(u, v) > 0 && graph(u, v) < Inf && ~visited(v)
            % 计算从起点经过u到v的距离
            new_distance = distances(u) + graph(u, v);
            % 如果新距离更小，则更新
            if new_distance < distances(v)
                distances(v) = new_distance;
                predecessors(v) = u;
            end
        end
    end
end

% 重建最短路径
paths = cell(1, num_nodes);
for end_node = 1:num_nodes
    path = [];
    current = end_node;
    % 从终点回溯到起点
    while ~isnan(current)
        path = [current, path];  % 在路径前添加当前节点
        if current == start_node
            break;
        end
        current = predecessors(current);
    end
    % 检查路径是否有效
    if isempty(path) || path(1) ~= start_node
        paths{end_node} = [];  % 不可达
    else
        paths{end_node} = path;
    end
end

% 输出结果
fprintf('\n最短路径结果：\n');
for i = 1:num_nodes
    if i == start_node
        continue;
    end
    if isempty(paths{i})
        fprintf('从%s到%s: 不可达\n', node_names{start_node}, node_names{i});
    else
        % 将节点索引转换为名称
        path_names = cell(1, length(paths{i}));
        for j = 1:length(paths{i})
            path_names{j} = node_names{paths{i}(j)};
        end
        fprintf('从%s到%s: 路径: %s, 距离: %.0f\n', ...
            node_names{start_node}, node_names{i}, ...
            strjoin(path_names, ' -> '), distances(i));
    end
end

% 可视化图和最短路径
figure('Position', [100 100 800 600]);

% 定义节点位置（使用圆形布局）
theta = linspace(0, 2*pi, num_nodes+1);
theta = theta(1:end-1);  % 7个角度
radius = 2;
x = radius * cos(theta);
y = radius * sin(theta);

% 绘制所有节点和边
hold on;
% 绘制所有边（灰色）
for i = 1:num_nodes
    for j = i+1:num_nodes
        if graph(i, j) > 0 && graph(i, j) < Inf
            plot([x(i) x(j)], [y(i) y(j)], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
        end
    end
end

% 绘制最短路径（红色高亮显示）
for i = 1:num_nodes
    if i ~= start_node && ~isempty(paths{i}) && length(paths{i}) > 1
        for j = 1:length(paths{i})-1
            u = paths{i}(j);
            v = paths{i}(j+1);
            plot([x(u) x(v)], [y(u) y(v)], 'r-', 'LineWidth', 2);
        end
    end
end

% 绘制节点
plot(x, y, 'bo', 'MarkerSize', 20, 'LineWidth', 2);
% 突出显示起点
plot(x(start_node), y(start_node), 'go', 'MarkerSize', 22, 'LineWidth', 2);

% 标注节点名称
for i = 1:num_nodes
    text(x(i)+0.1, y(i)+0.1, node_names{i}, 'FontSize', 12);
end

% 标注边权重
for i = 1:num_nodes
    for j = i+1:num_nodes
        if graph(i, j) > 0 && graph(i, j) < Inf
            mid_x = (x(i) + x(j)) / 2;
            mid_y = (y(i) + y(j)) / 2;
            text(mid_x, mid_y, num2str(graph(i, j)), 'FontSize', 10, ...
                'BackgroundColor', 'w', 'EdgeColor', 'none');
        end
    end
end

title(sprintf('Dijkstra算法最短路径（从%s出发）', node_names{start_node}), 'FontSize', 14);
axis equal;
axis off;
hold off;
    