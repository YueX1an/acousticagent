% verify_with_python_data.m
clc; clear; close all;

% --- 1. 读取 Python 生成的数据 ---
disp('正在读取 Python 生成的数据文件...');

if ~exist('matlab_input_params.txt', 'file') || ~exist('matlab_comparison_curves.txt', 'file')
    error('❌ 找不到 .txt 文件！请先运行 Python 脚本 step2_inverse_peak_focus_v2.py');
end

% 读取参数
sl_opt = load('matlab_input_params.txt');
sl_opt = sl_opt(:).'; % 转为行向量

% 读取曲线 (Col 1: Target, Col 2: AI_Pred)
curves = load('matlab_comparison_curves.txt');
y_target_python = curves(:, 1);
y_ai_pred = curves(:, 2);

% --- 2. 设置物理参数 (需与 generate_31param_data.m 一致) ---
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
H = 49.4; W = 49.4; mnum = 500; Norder = 9;

% 频率设置 (这里假设是 60-3500 里的前100个点，请根据你实际的 freq_axis 调整)
% 如果你的数据是均匀采样的 100 个点，请使用 linspace
fmin = 60; fmax = 2000; % 假设你之前的图是画到 2000Hz
f_axis = linspace(fmin, fmax, 100); 

% --- 3. 调用 MATLAB 物理引擎计算 ---
disp('🚀 正在调用 TMM 进行物理计算...');
[alpha_matlab, ~] = fun_structure_broadband_nlayer_HB_func(f_axis, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

% --- 4. 绘图对比 ---
figure('Color', 'w', 'Position', [100, 100, 1000, 600]);

% 画出 Python 里的 Target (绿色)
plot(f_axis, y_target_python, 'g-', 'LineWidth', 3, 'DisplayName', 'Target (Ground Truth)');
hold on;

% 画出 Python 里的 AI 预测 (红色虚线)
plot(f_axis, y_ai_pred, 'r--', 'LineWidth', 2, 'DisplayName', 'AI Prediction (Python)');

% 画出 MATLAB 算出的真实物理曲线 (蓝色圆圈)
plot(f_axis, alpha_matlab, 'bo', 'MarkerSize', 6, 'DisplayName', 'MATLAB Verification (TMM)');

title('Ultimate Verification: AI vs Physics Engine');
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
legend('Location', 'best');
grid on;
hold off;

% --- 5. 计算误差 ---
mse_check = mean((y_ai_pred - alpha_matlab').^2);
fprintf('\n🧐 验证报告:\n');
fprintf('   AI 预测 vs MATLAB 计算 MSE: %.8f\n', mse_check);

if mse_check < 1e-4
    disp('✅ 完美通过！AI 的预测完全符合物理规律！');
else
    disp('⚠️ 存在偏差，可能是频率轴 (Frequency Axis) 对齐有问题，请检查 f_axis 设置。');
end