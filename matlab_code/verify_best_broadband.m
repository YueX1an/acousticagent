% verify_best_broadband_v2.m
% 功能：验证 AI 预测的宽带吸声样本，并导出物理真值数据
clc; clear; close all;

fprintf('--- 🚀 正在启动顶刊级宽带吸声验证程序 ---\n');

%% ===================== 1. 配置区域 =====================
config = struct();
config.txt_folder = 'txt';          % 数据存放文件夹
config.save_data  = true;           % 是否导出 MATLAB 验证点信息
config.f_range    = [60, 2000];     % 频率扫描范围 (Hz)
config.f_pts      = 100;            % 采样点数
config.target_val = 1.0;            % 理想吸收目标值
config.eval_start = 500;            % 平均值评估起始频率 (Hz)

%% ===================== 2. 数据加载 =====================
% 拼接路径并检查文件
path_params = fullfile(config.txt_folder, 'matlab_input_params_best_broadband.txt');
path_curves = fullfile(config.txt_folder, 'matlab_curves_best_broadband.txt');

if ~exist(path_params, 'file') || ~exist(path_curves, 'file')
    error('❌ 关键数据文件丢失，请检查 txt 文件夹中的输入文件！');
end

% 加载 AI 预测的 31 维结构参数 [cite: 98, 531]
sl_opt = load(path_params);
sl_opt = sl_opt(:).'; 

% 加载 Python 传来的曲线数据 (Column 1: Target, Column 2: AI Predicted)
curves_data = load(path_curves);
y_target_py = curves_data(:, 1);
y_ai_py     = curves_data(:, 2);

%% ===================== 3. 物理引擎验证 (TMM) =====================
% 设定频率轴 [cite: 521]
f_axis = linspace(config.f_range(1), config.f_range(2), config.f_pts);

% 结构参数定义 [cite: 194, 525]
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
H = 49.4; W = 49.4; mnum = 500; Norder = 9;

fprintf('🔍 正在调用 TMM 物理算子进行严格验证...\n');
% 调用底层声学传递矩阵函数进行计算 [cite: 532]
[alpha_matlab, ~] = fun_structure_broadband_nlayer_HB_func(f_axis, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

%% ===================== 4. 数据导出 (新增功能) =====================
if config.save_data
    export_file = fullfile(config.txt_folder, 'matlab_verification_curve_best_broadband.txt');
    % 拼接频率和物理真值 [f, alpha]
    export_matrix = [f_axis(:), alpha_matlab(:)];
    save(export_file, 'export_matrix', '-ascii');
    fprintf('💾 物理验证点数据已保存至: %s\n', export_file);
end

%% ===================== 5. 制图渲染 =====================
figure('Color', 'w', 'Position', [200, 200, 950, 550]);
hold on;

% 计算 500-2000Hz 的平均吸声率 [cite: 189, 363]
eval_idx = f_axis >= config.eval_start;
avg_matlab = mean(alpha_matlab(eval_idx));
avg_ai = mean(y_ai_py(eval_idx));

% (A) 绘制目标理想区域 (Target Zone) [cite: 407]
patch_x = [config.eval_start, config.f_range(2), config.f_range(2), config.eval_start];
patch_y = [0, 0, 1.05, 1.05];
patch(patch_x, patch_y, [0.9 1.0 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Target Zone (1.0)');

% (B) 绘制 1.0 完美吸声参考线
yline(0.97, 'b:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% (C) 绘制 MATLAB 物理验证点 (蓝色空心圆圈) [cite: 341]
plot(f_axis, alpha_matlab, 'bo', 'LineWidth', 1.5, 'MarkerSize', 7, ...
     'DisplayName', sprintf('MATLAB Physics (Avg: %.3f)', avg_matlab));

% (D) 绘制 AI 预测曲线 (红色虚线) [cite: 46]
plot(f_axis, y_ai_py, 'r--', 'LineWidth', 2.5, ...
     'DisplayName', sprintf('AI Predicted (Avg: %.3f)', avg_ai));

% (E) 图表美化设置
title('Broadband Absorption Performance Verification', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Frequency (Hz)', 'FontSize', 14);
ylabel('Absorption Coefficient \alpha', 'FontSize', 14);

% 设置坐标轴属性
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'TickDir', 'out');
grid on;
set(gca, 'GridAlpha', 0.3);
xlim([0, 2000]);
ylim([0, 1.1]);

% 动态标注平均吸声率标签
text(200, 0.1, sprintf('Broadband Mean Abs: %.4f', avg_matlab), ...
     'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b', 'BackgroundColor', 'w');

legend('Location', 'northeast', 'FontSize', 11, 'Box', 'on');
hold off;

fprintf('✅ 验证完成！图片与数据均已就绪。\n');