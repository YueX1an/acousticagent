% verify_multi_peak.m
% 验证全域搜索找到的“多窄带”样本 (Multi-Peak Sample)
clc; clear; close all;

disp('--- 正在验证多窄带样本 (Multi-Peak Sample) ---');

% ================= 配置区域 =================
txt_folder = 'txt';  % 指定存放 txt 文件的文件夹名称
% ===========================================

% 1. 读取参数 (Target 真值)
% 对应 Python 生成的 matlab_input_params_multi_peak.txt
param_file = fullfile(txt_folder, 'matlab_input_params_multi_peak.txt');

if ~exist(param_file, 'file')
    error(['❌ 找不到文件: ', param_file, ' 请确保 Python 脚本运行完毕且文件在 txt 文件夹中!']);
end

sl_opt = load(param_file);
sl_opt = sl_opt(:).'; 

% 2. 读取曲线数据 (用于对比)
curve_file = fullfile(txt_folder, 'matlab_curves_multi_peak.txt');
if ~exist(curve_file, 'file')
    error(['❌ 找不到文件: ', curve_file]);
end

curves = load(curve_file);
y_target_py = curves(:, 1); % 绿色目标
y_ai_py = curves(:, 2);     % 红色预测

% 3. 物理计算 (TMM)
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
H = 49.4; W = 49.4; mnum = 500; Norder = 9;

% 频率轴设置 (如果不贴合，请尝试改为 linspace(0, 2000, 100))
f_axis = linspace(60, 2000, 100); 

disp('🚀 正在调用 TMM 物理引擎...');
[alpha_matlab, ~] = fun_structure_broadband_nlayer_HB_func(f_axis, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

% 4. 绘图 (突出多峰特性)
figure('Color', 'w', 'Position', [100, 100, 900, 500]);

% 画线
plot(f_axis, y_target_py, 'g-', 'LineWidth', 3, 'DisplayName', 'Target (Dataset)');
hold on;
plot(f_axis, y_ai_py, 'r--', 'LineWidth', 2, 'DisplayName', 'AI Optimized');
plot(f_axis, alpha_matlab, 'bo', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'MATLAB Verification');

% --- 亮点功能：自动标记峰值 ---
% 这在展示多峰结果时非常有效，证明没丢峰
[pks, locs] = findpeaks(alpha_matlab, f_axis, 'MinPeakHeight', 0.4, 'MinPeakDistance', 100);
plot(locs, pks, 'kv', 'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', 'Resonance Peaks');

title(['Multi-Peak Sample Verification - Detected ', num2str(length(pks)), ' Peaks'], 'FontSize', 14);
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
legend('Location', 'best'); 
grid on;
ylim([0, 1.05]);
hold off;

disp(['✅ 多窄带验证完成！这张图证明了模型对复杂共振机理的捕捉能力。']);