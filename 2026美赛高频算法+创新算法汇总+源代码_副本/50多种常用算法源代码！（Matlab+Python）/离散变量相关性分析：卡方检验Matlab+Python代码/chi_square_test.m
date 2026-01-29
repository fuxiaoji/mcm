% 卡方检验案例说明：
% 本案例使用卡方检验分析两个离散变量之间的相关性。卡方检验通过比较观测频数
% 与期望频数的差异来判断两个分类变量是否独立。值越大，表明两个变量相关性越强。
% 示例中使用泰坦尼克号数据集，分析"性别"与"生存情况"两个离散变量的相关性。

% 检查是否有统计工具箱
if ~license('test', 'Statistics_and_Machine_Learning_Toolbox')
    error('需要Statistics and Machine Learning Toolbox才能运行卡方检验');
end

% 创建泰坦尼克号数据集（性别与生存情况）
% 数据说明：1=男性，2=女性；0=未生存，1=生存
gender = [1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2;2];
survived = [0;0;0;0;0;0;0;0;0;1;0;0;0;0;0;1;1;1;1;1;1;1;1;0;1;1;1;1;1;0;1;1];

% 创建列联表
[contingency_table, gender_labels, survived_labels] = crosstab(gender, survived);

% 显示列联表
fprintf('列联表（性别 vs 生存情况）:\n');
disp(contingency_table);
fprintf('解释：行表示性别（1=男性，2=女性），列表示生存情况（0=未生存，1=生存）\n\n');

% 执行卡方检验
[chi2, p_value, dof, expected] = chi2test(contingency_table);

% 输出检验结果
fprintf('卡方检验结果：\n');
fprintf('卡方统计量: %.4f\n', chi2);
fprintf('P值: %.8f\n', p_value);
fprintf('自由度: %d\n', dof);
fprintf('\n期望频数表:\n');
disp(round(expected, 2));

% 结果解释
alpha = 0.05;
if p_value < alpha
    conclusion = sprintf('由于P值(%.8f) < %.2f，拒绝原假设，表明性别与生存情况显著相关。', p_value, alpha);
else
    conclusion = sprintf('由于P值(%.8f) ≥ %.2f，不拒绝原假设，表明性别与生存情况无显著相关。', p_value, alpha);
end
fprintf('\n结论：%s\n', conclusion);

% 可视化列联表数据
figure;
bar(contingency_table', 'stacked', 'FaceColor', 'flat');
set(gca, 'XTickLabel', {'男性', '女性'});
legend('未生存', '生存', 'Location', 'best');
title('泰坦尼克号乘客性别与生存情况的关系');
xlabel('性别');
ylabel('人数');
grid on;
text(1.5, max(sum(contingency_table,2))*0.9, ...
    sprintf('卡方值: %.2f, P值: %.6f', chi2, p_value), ...
    'HorizontalAlignment', 'center', 'BackgroundColor', 'w');
    