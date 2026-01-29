% BP神经网络分类模型案例说明：
% BP（反向传播）神经网络是一种多层前馈神经网络，通过反向传播算法调整网络权重来最小化预测误差。
% 它包含输入层、隐藏层和输出层，各层神经元之间全连接。
%
% 算法步骤：
% 1. 初始化网络权重和偏置
% 2. 前向传播：计算输入经过各层后的输出
% 3. 计算损失：比较预测输出与真实标签的差异
% 4. 反向传播：计算损失对各层权重的梯度
% 5. 更新权重：使用梯度下降法调整权重以减小损失
% 6. 重复步骤2-5直到收敛或达到最大迭代次数
%
% 本案例使用乳腺癌数据集，构建一个含1个隐藏层的BP神经网络进行二分类，
% 预测肿瘤是良性还是恶性，并评估模型性能。

% 检查神经网络工具箱
if ~license('test', 'Neural_Network_Toolbox')
    error('需要Neural Network Toolbox才能运行BP神经网络');
end

% 加载数据集
load breastcancer.mat;
X = breastcancer(:, 1:end-1);  % 特征数据（30个特征）
y = breastcancer(:, end);      % 标签（0=恶性，1=良性）
class_names = {'恶性', '良性'};

% 数据划分：训练集和测试集（7:3）
rng(42);  % 设置随机种子
cv = cvpartition(y, 'HoldOut', 0.3);  % 保留30%作为测试集
X_train = X(cv.training, :);
y_train = y(cv.training, :);
X_test = X(cv.test, :);
y_test = y(cv.test, :);

% 数据标准化（神经网络对数据尺度敏感）
[X_train_scaled, mu, sigma] = zscore(X_train);  % 训练集标准化
X_test_scaled = (X_test - mu) ./ sigma;  % 测试集标准化

% 构建BP神经网络模型
% 输入层：30个神经元（对应30个特征）
% 隐藏层：32个神经元，使用ReLU激活函数
% 输出层：1个神经元，使用sigmoid激活函数（二分类）
input_size = size(X_train_scaled, 2);
hidden_size = 32;
output_size = 1;

% 创建网络
net = feedforwardnet(hidden_size);
net.layers{1}.transferFcn = 'relu';  % 隐藏层激活函数
net.layers{2}.transferFcn = 'logsig';  % 输出层激活函数（sigmoid）

% 设置训练参数
net.trainParam.epochs = 300;  % 最大迭代次数
net.trainParam.lr = 0.001;    % 学习率
net.trainParam.goal = 0.01;   % 目标误差
net.trainParam.showWindow = false;  % 不显示训练窗口

% 训练模型
fprintf('训练BP神经网络...\n');
net = train(net, X_train_scaled', y_train');  % 注意输入为行向量

% 预测
y_pred_scores = net(X_test_scaled');  % 预测得分（0-1之间）
y_pred = (y_pred_scores >= 0.5);  % 以0.5为阈值转换为类别
y_pred = y_pred';  % 转换为列向量

% 输出评估结果
fprintf('\nBP神经网络在测试集上的性能：\n');
accuracy = mean(y_pred == y_test);
fprintf('准确率: %.4f\n', accuracy);

fprintf('\n混淆矩阵：\n');
conf_mat = confusionmat(y_test, y_pred);
disp(conf_mat);

fprintf('\n分类报告：\n');
disp(classification_report(y_test, y_pred));

% 绘制训练损失曲线
figure('Position', [100 100 800 500]);
plot(net.trainError.history);
xlabel('迭代次数');
ylabel('训练损失');
title('BP神经网络训练损失曲线');
grid on;

% 使用PCA降维到2D，可视化分类结果
[coeff, score, latent] = pca(X_train_scaled);
X_train_pca = score(:, 1:2);  % 取前两个主成分
X_test_pca = (X_test_scaled - mu) * coeff(:, 1:2);  % 转换测试集到PCA空间

% 绘制决策边界
h = 0.02;  % 网格步长
x_min = min(X_train_pca(:, 1)) - 1;
x_max = max(X_train_pca(:, 1)) + 1;
y_min = min(X_train_pca(:, 2)) - 1;
y_max = max(X_train_pca(:, 2)) + 1;
[xx, yy] = meshgrid(x_min:h:x_max, y_min:h:y_max);

% 使用PCA转换后的训练数据重新训练模型（仅用于可视化）
net_pca = feedforwardnet(hidden_size);
net_pca.layers{1}.transferFcn = 'relu';
net_pca.layers{2}.transferFcn = 'logsig';
net_pca.trainParam.epochs = 300;
net_pca.trainParam.showWindow = false;
net_pca = train(net_pca, X_train_pca', y_train');

Z = net_pca([xx(:) yy(:)]');
Z = (Z >= 0.5);
Z = reshape(Z, size(xx));

% 绘制ROC曲线
[y_pred_scores_all, ~] = net(X_test_scaled');
y_pred_proba = y_pred_scores_all';  % 正类的预测概率
[fpr, tpr, ~] = roc(y_test, y_pred_proba);
roc_auc = trapz(fpr, tpr);

% 可视化结果
figure('Position', [100 100 1000 500]);

% 1. 决策边界和样本点
subplot(1, 2, 1);
gscatter(xx(:), yy(:), Z, [], '.', 1);  % 决策区域
hold on;
% 绘制真实标签（方形）和预测标签（圆形）
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_test, ...
    'rb', 'ss', 10, 'off');  % 真实标签
gscatter(X_test_pca(:, 1), X_test_pca(:, 2), y_pred, ...
    'rb', 'oo', 6, 'off');  % 预测标签
title('BP神经网络决策边界 - 测试集');
xlabel(sprintf('PCA特征1 (%.2f%%)', latent(1)/sum(latent)*100));
ylabel(sprintf('PCA特征2 (%.2f%%)', latent(2)/sum(latent)*100));
legend([class_names; {'真实标签'; '预测标签'}], 'Location', 'best');
grid on;
hold off;

% 2. ROC曲线
subplot(1, 2, 2);
plot(fpr, tpr, 'b-', 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);  % 随机猜测的基准线
title(sprintf('ROC曲线 (面积 = %.4f)', roc_auc));
xlabel('假正例率 (FPR)');
ylabel('真正例率 (TPR)');
xlim([0, 1]);
ylim([0, 1.05]);
legend('ROC曲线', '随机猜测', 'Location', 'lower right');
grid on;
hold off;
    