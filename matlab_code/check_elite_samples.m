% =========================================================================
% evaluate_elite_samples.m
% 精英样本集的【物理合法性复核】与【宽带吸声性能并行评估】
% =========================================================================
clc; clear; close all;

% --- 1. 加载精英样本数据 ---
filename = 'elite_slN_1000.txt';
if ~isfile(filename)
    error(['找不到文件 ', filename, '，请确认是否已生成。']);
end
slN_elite = load(filename);
[N_elite, num_params] = size(slN_elite);
fprintf('成功读取 %d 个精英样本，准备进行终极质检...\n', N_elite);

% --- 2. 设定仿真环境参数 ---
fmin = 50; fstep = 10; fmax = 2000;
f0 = fmin:fstep:fmax;
fL = length(f0);
nlayer = 2; ncell = 4; nw = 2; delta = 1.0;
mnum = 500; Norder = 9;

% 预分配结果矩阵
alphaM_elite = zeros(fL, N_elite);
ZM_elite = zeros(fL, N_elite);
mean_alphas = zeros(1, N_elite);
physical_valid = true(1, N_elite);

% --- 3. 步骤一：极速物理空间复核 ---
fprintf('--------------------------------------------------\n');
fprintf('步骤 1/2: 正在进行物理防穿模复核...\n');
tol = 1e-4; 
err_count = 0;
for i = 1:N_elite
    sl = slN_elite(i, :);
    w1 = sl(26); w2 = sl(27);
    h1 = sl(17); h2 = sl(18); h3 = sl(19); h4 = sl(20);
    
    % 检查 Z 轴厚度 (总厚 57)
    z_err = any((sl(21:24) + sl(28:31) + 3.0) - sl(25) > tol);
    % 检查 XY 耦合 (总高/宽 46.7)
    xy_err = abs((h1+h2)-46.7)>tol || abs((h3+h4)-46.7)>tol || abs((w1+w2)-46.7)>tol;
    
    if z_err || xy_err
        physical_valid(i) = false;
        err_count = err_count + 1;
    end
end
fprintf('物理复核完成。合法样本: %d / %d (非法拦截: %d)\n', N_elite - err_count, N_elite, err_count);

% --- 4. 步骤二：大规模并行声学仿真 ---
fprintf('--------------------------------------------------\n');
fprintf('步骤 2/2: 正在进行多核并行声学仿真 (可能需要几分钟)...\n');
tic;
% 使用 parfor 加速计算
parfor i = 1:N_elite
    if physical_valid(i)
        sl = slN_elite(i, :);
        [alpha, Z] = fun_structure_broadband_nlayer_HB_func(f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
        alphaM_elite(:, i) = alpha;
        ZM_elite(:, i) = Z;
    end
end
toc;

% --- 5. 性能排名与统计 ---
% 计算所有合法样本在 50-2000Hz 频段内的平均吸声系数
for i = 1:N_elite
    if physical_valid(i)
        mean_alphas(i) = mean(alphaM_elite(:, i));
    else
        mean_alphas(i) = -1; % 废弃样本垫底
    end
end
% 降序排列，找出王者
[sorted_alphas, sort_idx] = sort(mean_alphas, 'descend');
best_idx = sort_idx(1); % 记录最佳样本的索引

fprintf('--------------------------------------------------\n');
fprintf('=== 🏆 精英样本性能报告 ===\n');
fprintf('最高平均吸声系数: %.4f (样本 #%d)\n', sorted_alphas(1), best_idx);
fprintf('前 10%% 样本的平均吸声系数底线: %.4f\n', sorted_alphas(floor(N_elite*0.1)));
fprintf('中位数吸声系数: %.4f\n', median(mean_alphas(mean_alphas > 0)));

% --- 6. 保存高性能结果与可视化 ---
disp('正在保存经过验证的精英吸声系数矩阵 (elite_alphaM_1000.txt)...');
save("elite_alphaM_1000.txt", "alphaM_elite", "-ascii");
save("elite_sorted_alphas.txt", "sorted_alphas", "-ascii");

% 绘制最强样本的曲线与部分优秀样本的“性能带”
figure('Name', 'Elite Samples Performance', 'Position', [100, 100, 800, 500]);
hold on;
% 画出排名前 50 的样本曲线 (用浅灰色半透明显示“基因群”)
for k = 1:min(50, N_elite)
    idx = sort_idx(k);
    plot(f0, alphaM_elite(:, idx), 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 1);
end
% 画出第一名王者曲线 (红色加粗)
plot(f0, alphaM_elite(:, best_idx), 'r-', 'LineWidth', 2.5);
grid on;
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Absorption Coefficient', 'FontSize', 12);
title(sprintf('Top Elite Absorption Curves (Best Mean \\alpha = %.4f)', sorted_alphas(1)), 'FontSize', 14);
legend('Top 50 Elite Variants', 'Best Sample', 'Location', 'best');
ylim([0 1.1]);
hold off;

% --- 7. 保存最佳王者样本的详细参数与曲线数据 ---
fprintf('--------------------------------------------------\n');
disp('正在导出最佳样本的 31 维结构参数与频响曲线...');

% 1. 保存 31 维物理结构参数
best_sl_params = slN_elite(best_idx, :);
save("best_structure_params_31D.txt", "best_sl_params", "-ascii");

% 2. 导出频点对应的吸声系数 (带表头，制表符分隔，方便后续绘图)
best_curve_file = 'best_absorption_curve.txt';
fid = fopen(best_curve_file, 'w');
if fid ~= -1
    % 写入表头
    fprintf(fid, 'Frequency(Hz)\tAbsorption_Coefficient\n');
    % 提取最佳曲线数据
    best_alpha_curve = alphaM_elite(:, best_idx);
    % 遍历写入频点和对应吸收率
    for j = 1:fL
        fprintf(fid, '%.1f\t%.6f\n', f0(j), best_alpha_curve(j));
    end
    fclose(fid);
    fprintf('✅ 最佳样本 31 维参数已保存至: best_structure_params_31D.txt\n');
    fprintf('✅ 最佳样本吸声曲线点信息已保存至: %s\n', best_curve_file);
else
    warning('无法创建文件 %s，请检查写入权限。', best_curve_file);
end

disp('质检与评估全部结束！图表及最终数据均已生成。');