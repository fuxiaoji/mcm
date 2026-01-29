% 拉格朗日插值法案例说明：
% 本案例使用拉格朗日插值法对已知数据点进行插值，估算未知点的值。
% 拉格朗日插值是一种多项式插值方法，通过构造一组基函数来逼近原函数。
% 示例中使用sin(x)函数的部分采样点进行插值，展示插值效果。

% 生成样本数据
x_sample = [0, pi/6, pi/3, pi/2, 2*pi/3, 5*pi/6, pi];  % 采样点
y_sample = sin(x_sample);  % 对应函数值

% 定义插值点
x_interp = linspace(0, pi, 100);  % 需要插值的点
y_interp = zeros(size(x_interp));  % 存储插值结果

% 拉格朗日插值核心计算
n = length(x_sample);  % 样本点数量
for k = 1:length(x_interp)
    x = x_interp(k);
    y = 0;
    for i = 1:n
        % 计算拉格朗日基函数L_i(x)
        L = 1;
        for j = 1:n
            if j ~= i
                L = L * (x - x_sample(j)) / (x_sample(i) - x_sample(j));
            end
        end
        % 累加得到插值结果
        y = y + y_sample(i) * L;
    end
    y_interp(k) = y;
end

% 计算真实值用于对比
y_true = sin(x_interp);

% 输出部分插值结果
fprintf('拉格朗日插值法结果（部分）：\n');
fprintf('%10s %15s %15s %10s\n', 'x值', '插值结果', '真实值', '误差');
fprintf('%10s %15s %15s %10s\n', '-----', '----------', '------', '-----');
indices = [1, 20, 40, 60, 80, 100];
for i = indices
    x_val = x_interp(i);
    interp_val = y_interp(i);
    true_val = y_true(i);
    error = abs(interp_val - true_val);
    fprintf('%10.4f %15.6f %15.6f %10.6e\n', x_val, interp_val, true_val, error);
end

% 可视化插值结果
figure;
plot(x_interp, y_true, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_interp, y_interp, 'r--', 'LineWidth', 1.5);
plot(x_sample, y_sample, 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
xlabel('x');
ylabel('y');
title('拉格朗日插值法示例');
legend('真实函数 sin(x)', '拉格朗日插值', '样本点');
grid on;
hold off;
    