% verify_design_WITH_SAFETY_FACTOR.m
% 
% 描述:
%   【最终验证 - 安全因子校准版】
%   验证 V3 模型逆向设计结果，并引入 Safety Factor 修正系统性偏差。
%   
%   绘图内容：
%     1. Target (蓝色实线): 目标曲线
%     2. Experimental (蓝色中空圆圈): MATLAB 物理真值 (X_optimized)
%     3. NN Predicted (红色虚线): 经过 Safety Factor 校准后的预测曲线

clc; clear; close all;

% ==============================================================================
% --- 核心配置 (在这里修改安全因子) ---
% ==============================================================================
% 建议尝试 0.90 ~ 0.98 之间的数值
% 0.95 表示认为模型预测值比真实值高了约 5%
SAFETY_FACTOR = 0.95; 

N_TARGET_INDEX = 58930; 
MODEL_TAG = 'V3_BASELINE_FINAL'; 
% ==============================================================================

disp(['--- 开始对比验证 (Safety Factor: ' num2str(SAFETY_FACTOR) ') ---']);

% --- 1. 设置仿真参数 ---
nlayer = 2;         
ncell = 4;          
nw = 2;             
delta = 0.7;        
mnum = 500;         
Norder = 9;         
fmin = 60; 
fstep = 1; 
fmax = 3500;
f0 = fmin:fstep:fmax; % 频率轴 (3441 个点)

% --- 2. 加载 Python 预测数据 ---
% 【重要：请确保此处文件名正确】
predicted_alpha_file = 'inverse_design_results/GIANT_FINAL_58930_20251125_160857_Alpha_predicted.txt'; 

if ~exist(predicted_alpha_file, 'file')
    warning(['文件未找到: ' predicted_alpha_file]);
    warning('请修改脚本第 42 行，填入正确的文件名！');
    return;
end

alpha_pred_raw = load(predicted_alpha_file);
alpha_pred_raw = alpha_pred_raw(:); % 确保是列向量

% --- 【核心步骤】应用安全因子进行校准 ---
alpha_pred_calibrated = alpha_pred_raw * SAFETY_FACTOR;
disp(['✅ 成功加载预测曲线，并应用 Safety Factor: ' num2str(SAFETY_FACTOR)]);


% --- 3. 定义结构参数 (V3 Giant Final 最佳结果) ---
% X_optimized (Found): Cos Sim 0.9595, MAE 5.3695
sl_optimized = [ ...
    6.17538229, 10.53246592, 8.92306045, 3.86774333, 9.49737473, 17.63175784, ...
    11.53226088, 13.67975917, 9.52659388, 18.95614003, 16.10344944, 12.74042866, ...
    28.55505373, 9.66730076, 12.51329093, 7.57868741, 43.98916645, 38.95652238, ...
    37.23610459, 38.99004525, 22.8525228, 28.09597915, 27.78906766, 22.34245318, ...
    55.11913183, 15.47058013, 33.10876695, 6.7837053, 8.59060623, 5.77148154, ...
    17.21825251 ...
];

% X_true (Target): Index 58930
sl_true = [
    1.3004797, 13.870184, 4.736627, 7.3717046, 15.180055, 3.2560651, ...
    16.661295, 17.148949, 14.325593, 11.604725, 11.933081, 15.298676, ...
    12.224637, 6.4053454, 8.274482, 11.547902, 46.929478, 30.544374, ...
    44.598293, 44.801395, 25.97552, 28.937828, 28.542393, 19.706562, ...
    50.799553, 19.082054, 28.217945, 5.155556, 11.699104, 19.519005, ...
    5.188179
];

% --- 4. 运行物理仿真 ---
disp('正在运行 MATLAB 物理仿真...');
% 仿真优化结构 -> 得到 Experimental/Real 曲线
[alpha_real, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl_optimized, ncell, nw, delta, nlayer, mnum, Norder);

% 仿真目标结构 -> 得到 Target 曲线
[alpha_target, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl_true, ncell, nw, delta, nlayer, mnum, Norder);
disp('物理仿真完成。');

% --- 5. 定制化绘图 ---
disp('正在绘制对比图 (0-2500 Hz)...');

% 裁剪数据范围 (0 - 2500 Hz)
f_plot_max = 2500;
idx_plot = f0 <= f_plot_max;

f_sub = f0(idx_plot);
target_sub = alpha_target(idx_plot);
real_sub = alpha_real(idx_plot);
pred_sub = alpha_pred_calibrated(idx_plot); % 使用校准后的数据

% 创建画布
figure('Position', [200, 200, 900, 600], 'Color', 'w');
hold on;

% 1. 目标曲线 (Target) - 蓝色粗实线
plot(f_sub, target_sub, 'b-', 'LineWidth', 3, 'DisplayName', 'Target');

% 2. 神经网络预测 (NN Predicted, Calibrated) - 红色虚线
plot(f_sub, pred_sub, 'r--', 'LineWidth', 2.5, ...
    'DisplayName', ['NN Predicted (Safety Factor ' num2str(SAFETY_FACTOR) ')']);

% 3. 实验/真实值 (Experimental) - 中空蓝色圆圈，稀疏采样
sample_rate = 20;
idx_sample = 1:sample_rate:length(f_sub);

plot(f_sub(idx_sample), real_sub(idx_sample), 'bo', ...
    'MarkerSize', 7, ...
    'LineWidth', 1.5, ...
    'MarkerFaceColor', 'none', ... 
    'DisplayName', 'Experimental');

% 图表装饰
title(['Inverse Design with Safety Factor ' num2str(SAFETY_FACTOR)], 'FontSize', 16);
xlabel('Frequency (Hz)', 'FontSize', 14);
ylabel('Absorption Coefficient (Alpha)', 'FontSize', 14);
xlim([0, 2550]); 
ylim([-0.05, 1.1]);
grid on;
ax = gca;
ax.GridAlpha = 0.3; 
legend('show', 'Location', 'northeast', 'FontSize', 12);
box on;

hold off;
disp('--- 绘图完成 ---');