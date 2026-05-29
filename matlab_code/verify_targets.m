% verify_targets.m
% 闭环验证: 读取 Python 预测参数 → TMM 仿真 → 对比 Target → 绘图
% 运行前先执行 validate_targets.py 生成 eval_data.mat
clc; clear; close all;

%% ==================== 配置 ====================
EVAL_DIR = '..\eval_results';
MAT_FILE = fullfile(EVAL_DIR, 'eval_data.mat');

% 仿真参数 (与 generate_strategic_dataset.m 完全一致)
nlayer = 2;  ncell = 4;  nw = 2;  delta = 1.0;
mnum = 500;  Norder = 9;

%% ==================== 加载数据 ====================
fprintf('========== 加载预测数据 ==========\n');
if ~exist(MAT_FILE, 'file')
    error('找不到 %s。请先运行 validate_targets.py。', MAT_FILE);
end

data = load(MAT_FILE);
target_spectra = data.target_spectra;   % (N_targets, 196)
target_names_cell = data.target_names;  % cell array
freqs = data.freqs(:);                  % (196, 1)
fL = length(freqs);

% 转换为字符串数组
target_names = strings(length(target_names_cell), 1);
for i = 1:length(target_names_cell)
    target_names(i) = string(target_names_cell{i});
end

N_targets = size(target_spectra, 1);
fprintf('加载完成: %d 个 Target, %d 频点 (%d-%d Hz)\n', N_targets, fL, freqs(1), freqs(end));

% 自动检测所有 params_* 字段
model_fields = fieldnames(data);
model_fields = model_fields(startsWith(model_fields, 'params_'));
N_models = length(model_fields);

model_labels = strings(N_models, 1);
model_short = strings(N_models, 1);
for j = 1:N_models
    fname = model_fields{j};               % e.g. 'params_T00_lambda0.0'
    model_short(j) = erase(fname, 'params_');  % e.g. 'T00_lambda0.0'
    % 友好标签
    parts = split(model_short(j), '_');
    if contains(model_short(j), 'T00')
        model_labels(j) = 'Trial 0 (\lambda=0.0, best)';
    elseif contains(model_short(j), 'T02')
        model_labels(j) = 'Trial 2 (\lambda=0.1, tuned)';
    elseif contains(model_short(j), 'Phase1')
        model_labels(j) = 'Phase 1 (\lambda=0.1, manual)';
    else
        model_labels(j) = model_short(j);
    end
end

fprintf('检测到 %d 个模型:\n', N_models);
for j = 1:N_models
    fprintf('  [%d] %s\n', j, model_labels(j));
end

%% ==================== TMM 仿真 ====================
fprintf('\n========== TMM 仿真 (共 %d 组) ==========\n', N_targets * N_models);

% 预分配
simulated = cell(N_models, 1);  % 每个 cell: (N_targets, 196)
sim_failed = zeros(N_models, N_targets);

for j = 1:N_models
    params = data.(model_fields{j});  % (N_targets, 31)
    alphaM = zeros(N_targets, fL);

    for i = 1:N_targets
        sl = params(i, :);
        try
            [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
                freqs', sl, ncell, nw, delta, nlayer, mnum, Norder);
            alphaM(i, :) = alpha(:)';
        catch ME
            fprintf('  [%s] Target %d (%s): 仿真失败 — %s\n', ...
                model_short(j), i, target_names(i), ME.message);
            alphaM(i, :) = NaN(1, fL);
            sim_failed(j, i) = 1;
        end
    end

    simulated{j} = alphaM;
    n_fail = sum(sim_failed(j, :));
    if n_fail == 0
        fprintf('  [%s] ✅ 全部 %d 组仿真成功\n', model_short(j), N_targets);
    else
        fprintf('  [%s] ⚠️ %d/%d 组仿真失败\n', model_short(j), n_fail, N_targets);
    end
end

%% ==================== 误差计算 ====================
fprintf('\n========== 误差分析 ==========\n');

freq_idx_400  = round((400 - freqs(1)) / 10) + 1;
freq_idx_900  = round((900 - freqs(1)) / 10) + 1;
freq_idx_1400 = round((1400 - freqs(1)) / 10) + 1;
freq_idx_1900 = round((1900 - freqs(1)) / 10) + 1;
four_idx = [freq_idx_400, freq_idx_900, freq_idx_1400, freq_idx_1900];

fprintf('\n--- 全频段 MSE (per target × model) ---\n');
fprintf('%-35s', 'Target');
for j = 1:N_models
    fprintf('  %-18s', model_short(j));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 35 + N_models * 20));

for i = 1:N_targets
    fprintf('%-35s', target_names(i));
    for j = 1:N_models
        alpha_sim = simulated{j}(i, :);
        alpha_tgt  = target_spectra(i, :);
        valid = ~isnan(alpha_sim);
        if any(valid)
            mse_val = mean((alpha_sim(valid) - alpha_tgt(valid)).^2);
            fprintf('  %18.6f', mse_val);
        else
            fprintf('  %18s', 'FAIL');
        end
    end
    fprintf('\n');
end

fprintf('\n--- 四频点 MAE ---\n');
four_freq_labels = {'400Hz','900Hz','1400Hz','1900Hz'};
for fi = 1:4
    fprintf('\n  %s:\n', four_freq_labels{fi});
    fprintf('  %-35s', 'Target');
    for j = 1:N_models
        fprintf('  %-18s', model_short(j));
    end
    fprintf('\n');
    for i = 1:N_targets
        fprintf('  %-35s', target_names(i));
        for j = 1:N_models
            sim_val = simulated{j}(i, four_idx(fi));
            tgt_val = target_spectra(i, four_idx(fi));
            if ~isnan(sim_val)
                fprintf('  %14.4f (tgt=%.2f)', abs(sim_val - tgt_val), tgt_val);
            else
                fprintf('  %18s', 'FAIL');
            end
        end
        fprintf('\n');
    end
end

%% ==================== 大图: 所有 Target 面板 ====================
fprintf('\n========== 绘图 ==========\n');

% 为每个模型创建一个大图
for j = 1:N_models
    figure('Name', sprintf('闭环验证 — %s', model_short(j)), ...
           'Position', [50, 50, 1600, 900], 'Color', 'w');

    n_cols = 5;
    n_rows = ceil(N_targets / n_cols);

    for i = 1:N_targets
        subplot(n_rows, n_cols, i);

        alpha_sim = simulated{j}(i, :);
        alpha_tgt = target_spectra(i, :);

        hold on;
        % Target 曲线 (黑色虚线)
        plot(freqs, alpha_tgt, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Target');
        % 仿真结果 (彩色实线)
        plot(freqs, alpha_sim, 'b-', 'LineWidth', 2.0, 'DisplayName', 'Simulated');

        % 标记四频点
        for fi = 1:4
            xline(freqs(four_idx(fi)), ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
        end

        % 四频点 Target vs Sim 散点
        plot(freqs(four_idx), alpha_tgt(four_idx), 'ko', ...
             'MarkerSize', 10, 'LineWidth', 1.5, 'MarkerFaceColor', 'k');
        plot(freqs(four_idx), alpha_sim(four_idx), 'ro', ...
             'MarkerSize', 8, 'LineWidth', 1.5, 'MarkerFaceColor', 'r');

        xlabel('Frequency (Hz)');
        ylabel('Absorption');
        title(target_names(i), 'Interpreter', 'none', 'FontSize', 9);
        legend off;
        xlim([50, 2000]);
        ylim([-0.05, 1.05]);
        grid on;
        set(gca, 'FontSize', 8);
    end

    sgtitle(sprintf('Closed-Loop Verification: %s  |  %d Targets  |  Black dashed = Target,  Blue = Simulated,  Red dots = Sim at key freqs', ...
        model_labels(j), N_targets), 'FontSize', 14, 'FontWeight', 'bold');

    % 保存图片
    saveas(gcf, fullfile(EVAL_DIR, sprintf('verify_%s.png', model_short(j))));
    fprintf('  已保存: verify_%s.png\n', model_short(j));
end

%% ==================== 对比图: 3 个模型叠加 ====================
figure('Name', '模型对比 — 全部 Target', ...
       'Position', [100, 100, 1600, 900], 'Color', 'w');

colors = lines(N_models);

n_cols = 5;
n_rows = ceil(N_targets / n_cols);

for i = 1:N_targets
    subplot(n_rows, n_cols, i);

    hold on;
    % Target (黑色虚线)
    h_tgt = plot(freqs, target_spectra(i, :), 'k--', 'LineWidth', 2.5);

    % 每个模型的仿真结果
    h_models = zeros(N_models, 1);
    for j = 1:N_models
        alpha_sim = simulated{j}(i, :);
        h_models(j) = plot(freqs, alpha_sim, '-', 'LineWidth', 1.5, ...
                           'Color', colors(j, :));
    end

    xlabel('Frequency (Hz)');
    ylabel('Absorption');
    title(target_names(i), 'Interpreter', 'none', 'FontSize', 9);
    xlim([50, 2000]);
    ylim([-0.05, 1.05]);
    grid on;
    set(gca, 'FontSize', 8);

    if i == 1
        legend([h_tgt; h_models], ['Target'; model_labels], ...
               'Location', 'best', 'FontSize', 7);
    end
end

sgtitle(sprintf('Model Comparison: %d Models × %d Targets  |  Black dashed = Target', ...
    N_models, N_targets), 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(EVAL_DIR, 'verify_comparison.png'));
fprintf('  已保存: verify_comparison.png\n');

%% ==================== 总结表格 ====================
fprintf('\n========== 总结 ==========\n');

fprintf('\n全频段 MSE 汇总:\n');
fprintf('%-35s', 'Target');
for j = 1:N_models
    fprintf('  %-12s', model_short(j));
end
fprintf('  %-12s\n', 'Best Model');

for i = 1:N_targets
    fprintf('%-35s', target_names(i));
    best_mse = inf;
    best_model = '';
    for j = 1:N_models
        alpha_sim = simulated{j}(i, :);
        alpha_tgt  = target_spectra(i, :);
        valid = ~isnan(alpha_sim);
        if any(valid)
            mse_val = mean((alpha_sim(valid) - alpha_tgt(valid)).^2);
            fprintf('  %12.6f', mse_val);
            if mse_val < best_mse
                best_mse = mse_val;
                best_model = model_short(j);
            end
        else
            fprintf('  %12s', 'FAIL');
        end
    end
    fprintf('  %-12s\n', best_model);
end

% 按 target 类型分组统计
fprintf('\n按 Target 类型分组:\n');
groups = {'Task1', 'Task2', 'Task3'};
for g = 1:length(groups)
    grp_mask = contains(target_names, groups{g});
    if any(grp_mask)
        grp_idx = find(grp_mask);
        fprintf('  %s (%d targets):\n', groups{g}, length(grp_idx));
        for j = 1:N_models
            mse_list = zeros(length(grp_idx), 1);
            for k = 1:length(grp_idx)
                alpha_sim = simulated{j}(grp_idx(k), :);
                alpha_tgt  = target_spectra(grp_idx(k), :);
                valid = ~isnan(alpha_sim);
                if any(valid)
                    mse_list(k) = mean((alpha_sim(valid) - alpha_tgt(valid)).^2);
                else
                    mse_list(k) = NaN;
                end
            end
            fprintf('    %-20s: MSE = %.6f ± %.6f  [%.6f, %.6f]\n', ...
                model_short(j), mean(mse_list, 'omitnan'), std(mse_list, 'omitnan'), ...
                min(mse_list), max(mse_list));
        end
    end
end

fprintf('\n========== 验证完成 ==========\n');
fprintf('所有图片已保存到: %s\n', EVAL_DIR);
