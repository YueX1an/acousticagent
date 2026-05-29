% verify_perfect_challenge.m
% 验证 AI 设计的“完美吸声体” (Target=1.0)
clc; clear; close all;

disp('--- 正在验证极限挑战结果 (Perfect Absorption Challenge) ---');

% ================= 配置区域 =================
txt_folder = 'txt';  
% ===========================================

% 1. 读取参数
param_file = fullfile(txt_folder, 'matlab_input_params_perfect.txt');
if ~exist(param_file, 'file')
    error(['❌ 找不到文件: ', param_file]);
end
sl_opt = load(param_file);
sl_opt = sl_opt(:).'; 

% 2. 读取曲线数据
curve_file = fullfile(txt_folder, 'matlab_curves_perfect.txt');
if ~exist(curve_file, 'file')
    error(['❌ 找不到文件: ', curve_file]);
end
curves = load(curve_file);
y_target_py = curves(:, 1); % 理想目标 (1.0)
y_ai_py = curves(:, 2);     % AI 预测 (0.976)

% 3. 物理计算 (TMM)
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
H = 49.4; W = 49.4; mnum = 500; Norder = 9;
f_axis = linspace(60, 2000, 100); 

disp('🚀 正在调用 TMM 物理引擎...');
[alpha_matlab, ~] = fun_structure_broadband_nlayer_HB_func(f_axis, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

% 4. 计算 500Hz 后的平均吸声率 (MATLAB版)
idx_500 = 23; 
avg_abs_matlab = mean(alpha_matlab(idx_500:end));

% 5. 绘图
figure('Color', 'w', 'Position', [100, 100, 900, 500]);

% 绘制“完美目标”区域
area(f_axis(idx_500:end), ones(size(f_axis(idx_500:end))), ...
    'FaceColor', [0.9 1 0.9], 'EdgeColor', 'none', ...
    'DisplayName', 'Target Zone (1.0)');
hold on;

% --- 绘制 AI 预测 (保持红色虚线，代表理论/预测) ---
plot(f_axis, y_ai_py, 'r--', 'LineWidth', 2, ...
    'DisplayName', ['AI Predicted (Avg: ', num2str(mean(y_ai_py(25:end)), '%.3f'), ')']);

% --- [修改处] 绘制 MATLAB 物理验证 (蓝色圆圈，模拟实验散点) ---
% 'bo' 表示蓝色圆圈 (Blue Circle) 且不连线
% MarkerSize 控制圆圈大小
% LineWidth 控制圆圈边缘的粗细
plot(f_axis, alpha_matlab, 'bo', 'MarkerSize', 5, 'LineWidth', 1.2, ...
    'DisplayName', ['MATLAB Physics (Avg: ', num2str(avg_abs_matlab, '%.3f'), ')']);

% 标记平均线
yline(avg_abs_matlab, 'b:', 'LineWidth', 2, 'HandleVisibility', 'off'); % HandleVisibility='off' 避免它出现在图例里

title(['Perfect Absorption Verification'], 'FontSize', 14);
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
legend('Location', 'southwest');
grid on;
ylim([0, 1.1]);
hold off;

fprintf('\n🏆 最终结果:\n');
fprintf('   AI 预测平均吸声率 (500-2000Hz): %.4f\n', mean(y_ai_py(25:end)));
fprintf('   MATLAB 物理验证平均吸声率:      %.4f\n', avg_abs_matlab);

if avg_abs_matlab > 0.9
    disp('🎉 奇迹！你真的设计出了一个超高性能的宽带吸声体！');
else
    disp('⚠️ 注意：物理验证结果略有下降，可能是频率轴对齐或物理极限原因。');
end