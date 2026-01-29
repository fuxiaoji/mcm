% Floyd算法案例说明：
% Floyd算法（弗洛伊德算法）是一种用于寻找图中所有节点对之间最短路径的动态规划算法。
% 与Dijkstra算法不同，它可以一次性计算出所有节点之间的最短路径，适用于有向图和无向图，
% 但不适用于包含负权回路的图。
%
% 算法步骤：
% 1. 初始化距离矩阵dist，dist(i,j)表示节点i到节点j的初始距离
% 2. 对于每个中间节点k（1到n）：
%    a. 对于每对节点(i,j)：
%       i. 计算从i经过k到j的距离：dist(i,k) + dist(k,j)
%       ii. 如果该距离小于当前dist(i,j)，则更新dist(i,j)
% 3. 算法结束后，dist(i,j)即为节点i到节点j的最短距离
%
% 本案例使用一个包含5个节点的有向图，演示Floyd算法如何计算所有节点对之间的最短路径，
% 并以矩阵形式展示结果，同时可视化部分关键路径。

% 定义图的邻接矩阵（表示节点间的距离，Inf表示无直接连接）
% 这是一个有向图，节点1-5代表5个不同的地点
graph = [
    0   3   Inf 7   Inf;   % 节点1到其他节点的距离
    8   0   2   Inf Inf;   % 节点2到其他节点的距离
    Inf Inf 0   1   4;     % 节点3到其他节点的距离
    Inf Inf Inf 0   5;     % 节点4到其他节点的距离
    Inf 6   Inf Inf 0      % 节点5到其他节点的距离
];

% 节点名称（用于可视化）
node_names = {'A', 'B', 'C', 'D', 'E'};
num_nodes = size(graph, 1);

% 初始化距离矩阵
dist_matrix = graph;
% 初始化路径矩阵，path(i,j)表示i到j的最短路径中j的前一个节点
path_matrix = zeros(num_nodes, num_nodes);
path_matrix(:) = NaN;

% 初始化路径矩阵：如果i和j直接相连，则path(i,j) = i
for i = 1:num_nodes
    for j = 1:num_nodes
        if i ~= j && graph(i, j) > 0 && graph(i, j) < Inf
            path_matrix(i, j) = i;
        end
    end
end

% Floyd算法核心
fprintf('使用Floyd算法计算所有节点对之间的最短路径...\n');
for k = 1:num_nodes  % 中间节点
    for i = 1:num_nodes  % 起点
        for j = 1:num_nodes  % 终点
            % 如果i到k和k到j都可达，且经过k的路径更短
            if dist_matrix(i, k) + dist_matrix(k, j) < dist_matrix(i, j)
                % 更新最短距离
                dist_matrix(i, j) = dist_matrix(i, k) + dist_matrix(k, j);
                % 更新路径：j的前一个节点是k的前一个节点
                path_matrix(i, j) = path_matrix(k, j);
            end
        end
    end
end

% 输出所有节点对之间的最短距离矩阵
fprintf('\n所有节点对之间的最短距离矩阵：\n');
% 打印表头
fprintf('    ');
for j = 1:num_nodes
    fprintf('%4s', node_names{j});
end
fprintf('\n');

% 打印矩阵内容
for i = 1:num_nodes
    fprintf('%s: ', node_names{i});
    for j = 1:num_nodes
        if dist_matrix(i, j) >= Inf
            fprintf('%4s', '∞');
        else
            fprintf('%4d', dist_matrix(i, j));
        end
    end
    fprintf('\n');
end

% 根据路径矩阵获取从start到end的最短路径
function path = get_path(path_matrix, start, end_node)
    if isnan(path_matrix(start, end_node))
        path = [];  % 不可达
        return;
    end
    % 重建路径
    path = end_node;
    current = end_node;
    while current ~= start
        current = path_matrix(start, current);
        if isnan(current)  % 路径不存在
            path = [];
            return;
        end
        path = [current, path];
    end
end

% 输出部分关键路径
fprintf('\n部分关键路径详情：\n');
key_pairs = [1 5; 2 4; 5 3];  % 选择几对有代表性的节点（起点 终点）
for i = 1:size(key_pairs, 1)
    start = key_pairs(i, 1);
    end_node = key_pairs(i, 2);
    path = get_path(path_matrix, start, end_node);
    if isempty(path)
        fprintf('从%s到%s: 不可达\n', node_names{start}, node_names{end_node});
    else
        % 将节点索引转换为名称
        path_names = cell(1, length(path));
        for j = 1:length(path)
            path_names{j} = node_names{path(j)};
        end
        fprintf('从%s到%s: 路径: %s, 距离: %d\n', ...
            node_names{start}, node_names{end_node}, ...
            strjoin(path_names, ' -> '), dist_matrix(start, end_node));
    end
end

% 可视化图和部分最短路径
figure('Position', [100 100 800 600]);

% 定义节点位置（使用圆形布局）
theta = linspace(0, 2*pi, num_nodes+1);
theta = theta(1:end-1);  % 5个角度
radius = 2;
x = radius * cos(theta);
y = radius * sin(theta);

% 绘制所有节点和边
hold on;

% 绘制所有边（灰色）
for i = 1:num_nodes
    for j = 1:num_nodes
        if graph(i, j) > 0 && graph(i, j) < Inf
            % 绘制有向边
            arrow_x = [x(i), x(j)];
            arrow_y = [y(i), y(j)];
            plot(arrow_x, arrow_y, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
            % 添加箭头
            annotation('arrow', ...
                [arrow_x(1) arrow_x(2)]/max(abs(arrow_x))/2+0.5, ...
                [arrow_y(1) arrow_y(2)]/max(abs(arrow_y))/2+0.5, ...
                'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'HeadWidth', 8);
        end
    end
end

% 绘制关键路径（红色高亮显示）
for i = 1:size(key_pairs, 1)
    start = key_pairs(i, 1);
    end_node = key_pairs(i, 2);
    path = get_path(path_matrix, start, end_node);
    if ~isempty(path) && length(path) > 1
        for j = 1:length(path)-1
            u = path(j);
            v = path(j+1);
            % 绘制有向边
            arrow_x = [x(u), x(v)];
            arrow_y = [y(u), y(v)];
            plot(arrow_x, arrow_y, 'r-', 'LineWidth', 2);
            % 添加箭头
            annotation('arrow', ...
                [arrow_x(1) arrow_x(2)]/max(abs(arrow_x))/2+0.5, ...
                [arrow_y(1) arrow_y(2)]/max(abs(arrow_y))/2+0.5, ...
                'Color', 'r', 'LineWidth', 2, 'HeadWidth', 8);
        end
    end
end

% 绘制节点
plot(x, y, 'bo', 'MarkerSize', 20, 'LineWidth', 2);

% 标注节点名称
for i = 1:num_nodes
    text(x(i)+0.1, y(i)+0.1, node_names{i}, 'FontSize', 12);
end

% 标注边权重
for i = 1:num_nodes
    for j = 1:num_nodes
        if graph(i, j) > 0 && graph(i, j) < Inf
            mid_x = (x(i) + x(j)) / 2;
            mid_y = (y(i) + y(j)) / 2;
            text(mid_x, mid_y, num2str(graph(i, j)), 'FontSize', 10, ...
                'BackgroundColor', 'w', 'EdgeColor', 'none');
        end
    end
end

title('Floyd算法计算的全源最短路径（红色为关键路径）', 'FontSize', 14);
axis equal;
axis off;
hold off;
    