% verify_ga_results.m
% 功能：审计 Genetic Algorithm (GA) 预测的结构参数，揭露其物理真实表现
% 特性：极度鲁棒的数据读取，兼容 Python 导出的各类型 txt 格式
clc; clear; close all;
fprintf('--- 🚀 正在启动 GA 结果物理审计程序 ---\n');

%% ===================== 1. 配置区域 =====================
config = struct();
config.txt_folder = 'txt';      % 指向包含 txt 文件的文件夹 (请确保此文件夹在当前工作路径下)
config.save_data  = true;           % 是否导出 MATLAB 验证点信息
config.f_range    = [50, 2000];     % 频率扫描范围 (Hz)
config.f_pts      = 100;            % 采样点数
config.eval_start = 50;             % 评估全频段平均吸声率

%% ===================== 2. 鲁棒的数据加载 (防错设计) =====================
path_params = fullfile(config.txt_folder, 'ga_best_parameters.txt');
path_curves = fullfile(config.txt_folder, 'ga_results.txt');

if ~exist(path_params, 'file') || ~exist(path_curves, 'file')
    error('❌ 找不到 GA 的数据文件！请检查 MATLAB 的当前工作路径 (Current Folder) 是否正确，且 %s 文件夹中存在相关文件。', config.txt_folder);
end

fprintf('📥 正在读取 GA 参数文件...\n');
% 核心升级：使用 readmatrix 自动跳过文本表头，只提取纯数值
try
    raw_params = readmatrix(path_params);
catch
    % 若 MATLAB 版本较老不支持 readmatrix，使用 importdata 作为后备方案
    temp_data = importdata(path_params);
    if isstruct(temp_data)
        raw_params = temp_data.data;
    else
        raw_params = temp_data;
    end
end

% 数据清洗：将一维数组展平，并清除可能因表头解析产生的 NaN (Not a Number)
raw_params = raw_params(:).';
raw_params = raw_params(~isnan(raw_params)); 

% 提取 MSE 和 31 维参数
ga_mse = raw_params(1);
sl_opt = raw_params(2:end);

% 终极维度核对
if length(sl_opt) ~= 31
    error('❌ 参数维度不匹配！期望 31 维，实际读取到 %d 维。请检查 txt 文件内容。', length(sl_opt));
end

fprintf('✅ 成功加载 GA 参数，Python 端报告 MSE 为: %.4f\n', ga_mse);

% 采用同样的鲁棒逻辑加载曲线数据
try
    curves_data = readmatrix(path_curves);
catch
    temp_curves = importdata(path_curves);
    if isstruct(temp_curves)
        curves_data = temp_curves.data;
    else
        curves_data = temp_curves;
    end
end

% 提取后立刻转置为行向量 (1 x 100)，迎合 TMM 底层算子的胃口
f_axis_py = curves_data(:, 1).'; 
y_ga_py   = curves_data(:, 2).';

%% ===================== 3. 物理引擎严格计算 (TMM) =====================
% 结构参数定义 (需与物理模型完全一致)
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
H = 49.4; W = 49.4; mnum = 500; Norder = 9;

fprintf('🔍 正在调用底层 TMM 算子进行残酷的物理验证...\n');
% 调用声学传递矩阵函数
[alpha_matlab, ~] = fun_structure_broadband_nlayer_HB_func(f_axis_py, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);

%% ===================== 4. 物理数据导出 (增强版) =====================
if config.save_data
    % 1. 导出验证曲线 [频率, 吸声率] -> 用于 Python 绘图
    % 文件名建议区分模型，如：matlab_val_PhyAR.txt 或 matlab_val_GA.txt
    curve_file = fullfile(config.txt_folder, 'matlab_val_points_GA.txt');
    curve_matrix = [f_axis_py(:), alpha_matlab(:)];
    save(curve_file, 'curve_matrix', '-ascii');
    
    % 2. 导出指标摘要 [MSE, Accuracy, Mean_Alpha] -> 用于汇总统计
    summary_file = fullfile(config.txt_folder, 'matlab_val_summary_GA.txt');
    % 计算该样本相对于目标 1.0 的指标
    val_mse = mean((alpha_matlab - 1.0).^2);
    val_acc = max(0, (1.0 - mean(abs(alpha_matlab - 1.0)))) * 100;
    val_avg = mean(alpha_matlab);
    
    summary_data = [val_mse, val_acc, val_avg];
    save(summary_file, 'summary_data', '-ascii');
    
    fprintf('💾 验证数据已保存：\n');
    fprintf('   - 曲线点: %s\n', curve_file);
    fprintf('   - 指标摘要: %s\n', summary_file);
end

%% ===================== 5. 顶刊级审计对比图 =====================
figure('Color', 'w', 'Position', [200, 200, 900, 500]);
hold on;

% 计算平均吸声率
avg_matlab = mean(alpha_matlab);
avg_ai     = mean(y_ga_py);

% (A) 绘制 1.0 完美吸声参考线
yline(1.0, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% (B) 绘制 MATLAB 物理真值点 (深绿色带圈实线) -> 代表真实的物理响应
plot(f_axis_py, alpha_matlab, '-o', 'Color', '#2ca02c', 'LineWidth', 1.5, ...
     'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
     'DisplayName', sprintf('MATLAB Ground Truth (Avg: %.3f)', avg_matlab));

% (C) 绘制 GA 在神经网络中的“幻想”预测曲线 (红色粗虚线)
plot(f_axis_py, y_ga_py, 'r--', 'LineWidth', 2.5, ...
     'DisplayName', sprintf('GA Predicted in NN (Avg: %.3f)', avg_ai));

% (D) 图表美化设置
title('Physical Audit of Genetic Algorithm (GA) Results', 'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'Arial');
xlabel('Frequency (Hz)', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Arial');
ylabel('Absorption Coefficient \alpha', 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Arial');

% 设置坐标轴属性
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'TickDir', 'in', 'FontName', 'Arial');
grid on;
set(gca, 'GridAlpha', 0.3, 'GridLineStyle', ':');
xlim([50, 2000]);
ylim([0, 1.15]); % 顶部留白，防止图例遮挡

% 动态标注对比差异框
diff_avg = abs(avg_matlab - avg_ai);
text_str = {sprintf('Python NN MSE: %.4f', ga_mse), ...
            sprintf('Reality vs NN Gap: %.4f', diff_avg)};
text(100, 0.15, text_str, 'FontSize', 12, 'FontWeight', 'bold', ...
     'Color', 'red', 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 4);

legend('Location', 'southeast', 'FontSize', 11, 'Box', 'on', 'EdgeColor', 'k');
hold off;

fprintf('✅ GA 物理审计完成！\n');