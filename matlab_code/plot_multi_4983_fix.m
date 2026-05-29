% plot_multi_4983_fix.m (最终版)
clc; clear; close all;

% ==================================================
% 【请修改】填入您刚刚生成的这个文件名 (不带后缀)
BASE_NAME = 'FIX_Multi_4983_20251208_140545'; 
% ==================================================

result_dir = 'inverse_design_results';
target_index = 4983;

% 仿真参数
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; mnum = 500; Norder = 9;
f0 = 60:1:3500; 

% 文件路径
x_opt_file = fullfile(result_dir, [BASE_NAME '_X_optimized.txt']);
x_true_file = fullfile(result_dir, [BASE_NAME '_X_true.txt']); 
pred_file = fullfile(result_dir, [BASE_NAME '_Alpha_pred.txt']);

if ~exist(x_opt_file, 'file') || ~exist(x_true_file, 'file')
    error(['找不到文件: ' BASE_NAME '。请检查文件名是否正确。']);
end

% 加载数据
sl_opt = load(x_opt_file);
sl_true = load(x_true_file);
alpha_pred = load(pred_file);

% 自动判断预测曲线的频率轴
if length(alpha_pred) == 100
    f_pred = linspace(0, 2000, 100);
else
    f_pred = linspace(60, 3500, length(alpha_pred));
end

% 运行仿真
disp('1. 仿真 Target (蓝线)...');
[alpha_target, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl_true, ncell, nw, delta, nlayer, mnum, Norder);

disp('2. 仿真 Physical (绿线)...');
[alpha_real, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

% 绘图
disp('3. 绘图...');
figure('Position', [100, 100, 800, 600], 'Color', 'w');
hold on;

idx_plot = f0 <= 2000;

% Target (蓝色实线)
plot(f0(idx_plot), alpha_target(idx_plot), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Target (Simulated)');

% Physical (绿色实线)
plot(f0(idx_plot), alpha_real(idx_plot), 'g-', 'LineWidth', 2, 'DisplayName', 'Physical (MATLAB)');

% Predicted (红色虚线)
plot(f_pred, alpha_pred, 'r--', 'LineWidth', 2, 'DisplayName', 'NN Predicted');

title(['Multi-Band Design (Index ' num2str(target_index) ')'], 'FontSize', 16);
xlabel('Frequency (Hz)', 'FontSize', 14);
ylabel('Absorption Coefficient', 'FontSize', 14);
ylim([-0.05, 1.1]);
grid on; ax = gca; ax.GridAlpha = 0.3;
legend('show', 'Location', 'best', 'FontSize', 12);
box on;

hold off;
disp('完成！请查看生成的图像。');