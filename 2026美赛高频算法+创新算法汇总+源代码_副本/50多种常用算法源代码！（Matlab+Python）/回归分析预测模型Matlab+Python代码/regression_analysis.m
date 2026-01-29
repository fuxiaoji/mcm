% 回归分析预测模型案例说明：
% 本案例使用多元线性回归和多项式回归模型预测房屋价格，
% 基于房屋面积、卧室数量和建造年份三个特征。回归分析
% 是一种统计方法，用于建立自变量和因变量之间的关系模型。
% 本案例比较线性回归和二次多项式回归的效果，展示如何
% 处理特征与目标之间的非线性关系。

% 1. 准备数据
% 设置随机种子，确保结果可复现
rng(42);

% 生成模拟数据：200套房屋的特征和价格
n_samples = 200;

% 特征数据：[房屋面积(平方米), 卧室数量, 建造年份]
area = 50 + (200-50)*rand(n_samples, 1);          % 房屋面积
bedrooms = randi([1, 4], n_samples, 1);            % 卧室数量
year_built = 1980 + (2020-1980)*rand(n_samples, 1); % 建造年份

% 组合特征
X = [area, bedrooms, year_built];

% 目标数据：房屋价格(万元)，与特征呈线性和非线性关系
% 价格公式：基础价格 + 面积*单价 + 卧室数量*加成 + 年份因素(非线性) + 随机噪声
price = 50 + 0.8*area + 10*bedrooms + 0.02*(year_built-2000).^2 + 8*randn(n_samples, 1);
price(price < 80) = 80;  % 确保价格不为过低

% 2. 数据预处理：标准化
[X_scaled, X_mu, X_sigma] = zscore(X);
[y_scaled, y_mu, y_sigma] = zscore(price);

% 3. 划分训练集和测试集
indices = randperm(n_samples);
train_ratio = 0.8;
train_size = round(train_ratio * n_samples);
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

X_train = X_scaled(train_indices, :);
y_train = y_scaled(train_indices);
X_test = X_scaled(test_indices, :);
y_test = y_scaled(test_indices);

% 4. 线性回归模型
% 创建并训练模型
linear_model = fitlm(X_train, y_train);

% 预测
y_train_pred_linear = predict(linear_model, X_train);
y_test_pred_linear = predict(linear_model, X_test);

% 5. 多项式回归模型（二次）
% 创建多项式特征（二次项和交互项）
X_train_poly = [X_train, X_train(:,1).^2, X_train(:,2).^2, X_train(:,3).^2, ...
                X_train(:,1).*X_train(:,2), X_train(:,1).*X_train(:,3), X_train(:,2).*X_train(:,3)];
X_test_poly = [X_test, X_test(:,1).^2, X_test(:,2).^2, X_test(:,3).^2, ...
               X_test(:,1).*X_test(:,2), X_test(:,1).*X_test(:,3), X_test(:,2).*X_test(:,3)];

% 创建并训练模型
poly_model = fitlm(X_train_poly, y_train);

% 预测
y_train_pred_poly = predict(poly_model, X_train_poly);
y_test_pred_poly = predict(poly_model, X_test_poly);

% 6. 反标准化预测结果
y_train_actual = y_train * y_sigma + y_mu;
y_test_actual = y_test * y_sigma + y_mu;
y_train_pred_linear_actual = y_train_pred_linear * y_sigma + y_mu;
y_test_pred_linear_actual = y_test_pred_linear * y_sigma + y_mu;
y_train_pred_poly_actual = y_train_pred_poly * y_sigma + y_mu;
y_test_pred_poly_actual = y_test_pred_poly * y_sigma + y_mu;

% 7. 模型评估
% 计算线性回归性能指标
linear_train_rmse = sqrt(mean((y_train_actual - y_train_pred_linear_actual).^2));
linear_test_rmse = sqrt(mean((y_test_actual - y_test_pred_linear_actual).^2));
linear_train_r2 = 1 - sum((y_train_actual - y_train_pred_linear_actual).^2) / sum((y_train_actual - mean(y_train_actual)).^2);
linear_test_r2 = 1 - sum((y_test_actual - y_test_pred_linear_actual).^2) / sum((y_test_actual - mean(y_test_actual)).^2);

% 计算多项式回归性能指标
poly_train_rmse = sqrt(mean((y_train_actual - y_train_pred_poly_actual).^2));
poly_test_rmse = sqrt(mean((y_test_actual - y_test_pred_poly_actual).^2));
poly_train_r2 = 1 - sum((y_train_actual - y_train_pred_poly_actual).^2) / sum((y_train_actual - mean(y_train_actual)).^2);
poly_test_r2 = 1 - sum((y_test_actual - y_test_pred_poly_actual).^2) / sum((y_test_actual - mean(y_test_actual)).^2);

% 8. 预测新数据
% 新房屋特征
new_houses = [
    100, 2, 2000;  % 100平米，2卧室，2000年建造
    150, 3, 2010;  % 150平米，3卧室，2010年建造
    80, 1, 1990    % 80平米，1卧室，1990年建造
];
% 标准化
new_houses_scaled = (new_houses - X_mu) ./ X_sigma;
% 创建多项式特征
new_houses_poly = [new_houses_scaled, new_houses_scaled(:,1).^2, new_houses_scaled(:,2).^2, new_houses_scaled(:,3).^2, ...
                   new_houses_scaled(:,1).*new_houses_scaled(:,2), new_houses_scaled(:,1).*new_houses_scaled(:,3), new_houses_scaled(:,2).*new_houses_scaled(:,3)];

% 预测价格
linear_pred = predict(linear_model, new_houses_scaled) * y_sigma + y_mu;
poly_pred = predict(poly_model, new_houses_poly) * y_sigma + y_mu;

% 9. 输出结果
disp("回归分析预测模型结果：");
disp("\n线性回归模型系数：");
disp(linear_model.Coefficients);

disp("\n模型性能比较：");
fprintf("线性回归 - 训练集RMSE: %.2f万元, R²: %.4f\n", linear_train_rmse, linear_train_r2);
fprintf("线性回归 - 测试集RMSE: %.2f万元, R²: %.4f\n", linear_test_rmse, linear_test_r2);
fprintf("二次多项式回归 - 训练集RMSE: %.2f万元, R²: %.4f\n", poly_train_rmse, poly_train_r2);
fprintf("二次多项式回归 - 测试集RMSE: %.2f万元, R²: %.4f\n", poly_test_rmse, poly_test_r2);

disp("\n新房屋价格预测：");
for i = 1:size(new_houses, 1)
    fprintf("%d平米, %d卧室, %.0f年建造:\n", new_houses(i,1), new_houses(i,2), new_houses(i,3));
    fprintf("  线性回归预测: %.2f万元\n", linear_pred(i));
    fprintf("  多项式回归预测: %.2f万元\n", poly_pred(i));
end

% 10. 可视化结果（房屋面积与价格的关系）
figure;
% 选择卧室数量=2，建造年份在2000年左右的样本点进行可视化
mask = (bedrooms == 2) & (year_built > 1995) & (year_built < 2005);
area_subset = area(mask);
price_subset = price(mask);

% 排序用于绘图
[sorted_area, sorted_idx] = sort(area_subset);
sorted_price = price_subset(sorted_idx);

% 创建用于预测的特征
X_plot = [sorted_area, 2*ones(size(sorted_area)), 2000*ones(size(sorted_area))];
% 标准化
X_plot_scaled = (X_plot - X_mu) ./ X_sigma;
% 创建多项式特征
X_plot_poly = [X_plot_scaled, X_plot_scaled(:,1).^2, X_plot_scaled(:,2).^2, X_plot_scaled(:,3).^2, ...
               X_plot_scaled(:,1).*X_plot_scaled(:,2), X_plot_scaled(:,1).*X_plot_scaled(:,3), X_plot_scaled(:,2).*X_plot_scaled(:,3)];

% 预测
linear_plot_pred = predict(linear_model, X_plot_scaled) * y_sigma + y_mu;
poly_plot_pred = predict(poly_model, X_plot_poly) * y_sigma + y_mu;

% 绘制
scatter(sorted_area, sorted_price, 50, 'blue', 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', '实际数据点');
hold on;
plot(sorted_area, linear_plot_pred, 'r-', 'LineWidth', 2, 'DisplayName', '线性回归');
plot(sorted_area, poly_plot_pred, 'g--', 'LineWidth', 2, 'DisplayName', '二次多项式回归');
xlabel('房屋面积 (平方米)');
ylabel('房屋价格 (万元)');
title('房屋面积与价格关系（控制变量：2卧室，2000年左右建造）');
legend();
grid on;
hold off;
