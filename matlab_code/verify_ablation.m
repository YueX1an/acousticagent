% verify_ablation.m
% 消融实验 TMM 仿真验证
% ======================================================================
% 读取 predict_ablation.py 生成的结构参数 → TMM 仿真 → 保存频谱 → 绘图
%
% 消融轴:
%   1. Remove SE attention:  PhyARNetNoSE vs PhyARNet (both 0.1)
%   2. Remove physics loss:  0.0 vs 0.1 (both PhyARNet with SE)
%
% 任务:
%   Task 1 — 宽带吸声 (txt/broadband/)
%   Task 2 — 1907Hz 全息图 (txt/hologram/)
%
% 使用前先运行: conda activate dnn && python predict_ablation.py
% ======================================================================
clc; clear; close all;

%% ==================== 配置 ====================
DATA_DIR = '..\txt';
TASKS = {'broadband', 'hologram'};
TASK_LABELS = {'Task1: Broadband Absorption', 'Task2: 1907Hz Hologram'};
DATE = '20260518';

% 3 个消融模型
MODEL_IDS = {'Phase1_lambda0.1', 'NoSE_lambda0.1', 'T00_lambda0.0'};
MODEL_LABELS = {
    'Phase1 0.1 (Full: SE + physics loss)',
    'NoSE 0.1 (Remove SE, keep physics loss)',
    'T00 0.0 (Remove physics loss, keep SE)'
};
MODEL_ABLATION = {'baseline', 'remove_attention', 'remove_physics_loss'};

% TMM 仿真参数 (与 generate_strategic_dataset.m 一致)
nlayer = 2;  ncell = 4;  nw = 2;  delta = 1.0;
mnum = 500;  Norder = 9;

% 频率轴
FMIN = 50;  FSTEP = 10;  FMAX = 2000;
freqs = double((FMIN:FSTEP:FMAX)');  % (196, 1), explicit double
fL = length(freqs);

% 四频点索引 (1-based)
idx_400  = round((400  - FMIN) / FSTEP) + 1;  % 36
idx_900  = round((900  - FMIN) / FSTEP) + 1;  % 86
idx_1400 = round((1400 - FMIN) / FSTEP) + 1;  % 136
idx_1900 = round((1900 - FMIN) / FSTEP) + 1;  % 186
four_idx = [idx_400, idx_900, idx_1400, idx_1900];
four_labels = {'400Hz', '900Hz', '1400Hz', '1900Hz'};

%% ==================== 遍历任务 ====================
for tk = 1:length(TASKS)
    task_dir = fullfile(DATA_DIR, TASKS{tk});
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('  %s\n', TASK_LABELS{tk});
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Data dir: %s\n', task_dir);

    % --- 自动发现该任务下的 target + param 文件 ---
    % Target 文件: {DATE}_target_*.txt
    target_files = dir(fullfile(task_dir, sprintf('%s_target_*.txt', DATE)));
    N_targets = length(target_files);

    if N_targets == 0
        fprintf('WARNING: No target files found in %s, skipping.\n', task_dir);
        continue;
    end

    % 加载所有 target spectra
    target_spectra = zeros(N_targets, fL);
    target_names = cell(N_targets, 1);
    for i = 1:N_targets
        fpath = fullfile(task_dir, target_files(i).name);
        % 跳过注释行 (# 开头)
        fid = fopen(fpath, 'r');
        data = [];
        while ~feof(fid)
            line = strtrim(fgetl(fid));
            if isempty(line) || startsWith(line, '#') || startsWith(line, '=')
                continue;
            end
            data = [data; str2double(line)];
        end
        fclose(fid);
        target_spectra(i, :) = data(:)';
        % 提取 task name (去掉日期和 target 前缀)
        tname = erase(target_files(i).name, sprintf('%s_target_', DATE));
        tname = erase(tname, '.txt');
        target_names{i} = tname;
        fprintf('  Target [%d]: %s (%d freqs)\n', i, tname, length(data));
    end

    % --- TMM 仿真每个模型 × 每个 target ---
    N_models = length(MODEL_IDS);
    simulated = cell(N_models, N_targets);
    sim_failed = zeros(N_models, N_targets);

    for m = 1:N_models
        fprintf('\n--- Model: %s ---\n', MODEL_LABELS{m});

        for i = 1:N_targets
            % 确定 target 短名用于文件名
            if contains(target_names{i}, 'Broadband')
                short_task = 'Task1_Broadband';
            elseif contains(target_names{i}, 'Target')
                short_task = 'Hologram_Target';
            elseif contains(target_names{i}, 'Background')
                short_task = 'Hologram_Background';
            else
                short_task = target_names{i};
            end

            param_fname = sprintf('%s_%s_%s.txt', DATE, MODEL_IDS{m}, short_task);
            param_fpath = fullfile(task_dir, param_fname);

            if ~exist(param_fpath, 'file')
                fprintf('  [%s] WARNING: %s not found, skipping.\n', target_names{i}, param_fname);
                sim_failed(m, i) = -1;
                continue;
            end

            % 读取参数 (跳过注释)
            fid = fopen(param_fpath, 'r');
            sl = [];
            while ~feof(fid)
                line = strtrim(fgetl(fid));
                if isempty(line) || startsWith(line, '#') || startsWith(line, '=')
                    continue;
                end
                sl = [sl; str2double(line)];
            end
            fclose(fid);
            sl = sl(:)';  % (1, 31)

            % TMM 仿真
            try
                [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
                    freqs', sl, ncell, nw, delta, nlayer, mnum, Norder);
                simulated{m, i} = alpha(:)';
                fprintf('  [%s] OK — α@400=%.3f, α@900=%.3f, α@1400=%.3f, α@1900=%.3f\n', ...
                    target_names{i}, alpha(idx_400), alpha(idx_900), alpha(idx_1400), alpha(idx_1900));
            catch ME
                fprintf('  [%s] FAIL — %s\n', target_names{i}, ME.message);
                simulated{m, i} = NaN(1, fL);
                sim_failed(m, i) = 1;
            end
        end
    end

    % --- 保存仿真结果 ---
    fprintf('\n--- Saving simulated spectra ---\n');
    for m = 1:N_models
        for i = 1:N_targets
            if sim_failed(m, i) ~= 0
                continue;
            end
            if contains(target_names{i}, 'Broadband')
                short_task = 'Task1_Broadband';
            elseif contains(target_names{i}, 'Target')
                short_task = 'Hologram_Target';
            else
                short_task = 'Hologram_Background';
            end

            sim_fname = sprintf('%s_%s_%s_simulated.txt', DATE, MODEL_IDS{m}, short_task);
            sim_fpath = fullfile(task_dir, sim_fname);

            alpha_save = simulated{m, i}(:);  % column vector
            fid = fopen(sim_fpath, 'w');
            fprintf(fid, '# Simulated absorption spectrum\n');
            fprintf(fid, '# Model: %s (%s)\n', MODEL_IDS{m}, MODEL_LABELS{m});
            fprintf(fid, '# Target: %s\n', target_names{i});
            fprintf(fid, '# Freq range: %d:%d:%d Hz (%d points)\n', FMIN, FSTEP, FMAX, fL);
            fprintf(fid, '# Ablation: %s\n', MODEL_ABLATION{m});
            fprintf(fid, '# Columns: absorption_coefficient\n');
            fprintf(fid, '%s\n', repmat('=', 50));
            for k = 1:fL
                fprintf(fid, '%.6f\n', alpha_save(k));
            end
            fclose(fid);
            fprintf('  %s\n', sim_fname);
        end
    end

    % --- 误差分析 ---
    fprintf('\n========== Error Analysis: %s ==========\n', TASK_LABELS{tk});

    % 全频段 MSE 表格
    fprintf('\n--- Full-Band MSE (per target x model) ---\n');
    fprintf('%-30s', 'Target');
    for m = 1:N_models
        fprintf('  %-20s', MODEL_IDS{m});
    end
    fprintf('  %-12s\n', 'Best Model');
    fprintf('%s\n', repmat('-', 1, 30 + N_models*22 + 14));

    for i = 1:N_targets
        fprintf('%-30s', target_names{i});
        best_mse = inf;
        best_m = 0;
        for m = 1:N_models
            if sim_failed(m, i) == 0
                alpha_sim = simulated{m, i};
                mse_val = mean((alpha_sim - target_spectra(i, :)).^2);
                fprintf('  %20.6f', mse_val);
                if mse_val < best_mse
                    best_mse = mse_val;
                    best_m = m;
                end
            else
                fprintf('  %20s', 'FAIL');
            end
        end
        if best_m > 0
            fprintf('  %-12s', MODEL_IDS{best_m});
        else
            fprintf('  %-12s', 'N/A');
        end
        fprintf('\n');
    end

    % 四频点 MAE
    fprintf('\n--- Per-Frequency MAE ---\n');
    for fk = 1:4
        fidx = four_idx(fk);
        fprintf('\n@%s:\n', four_labels{fk});
        fprintf('%-30s', 'Target');
        for m = 1:N_models
            fprintf('  %-12s', MODEL_IDS{m});
        end
        fprintf('\n');

        for i = 1:N_targets
            fprintf('%-30s', target_names{i});
            for m = 1:N_models
                if sim_failed(m, i) == 0
                    abs_err = abs(simulated{m, i}(fidx) - target_spectra(i, fidx));
                    fprintf('  %12.4f', abs_err);
                else
                    fprintf('  %12s', 'FAIL');
                end
            end
            fprintf('\n');
        end
    end

    % --- 对比绘图 ---
    fprintf('\n--- Generating plots ---\n');

    % 模型颜色
    colors = lines(N_models);

    for i = 1:N_targets
        figure('Position', [100, 100, 1200, 500]);

        % 子图1: 全频段频谱对比
        subplot(1, 2, 1);
        hold on;
        % 画 target
        h_target = plot(freqs, target_spectra(i, :), 'k--', 'LineWidth', 2, 'DisplayName', 'Target');
        h_models = zeros(N_models, 1);
        for m = 1:N_models
            if sim_failed(m, i) == 0
                h_models(m) = plot(freqs, simulated{m, i}, '-', ...
                    'Color', colors(m, :), 'LineWidth', 1.2, ...
                    'DisplayName', MODEL_IDS{m});
            end
        end
        % 标注四频点
        for fk = 1:4
            xline(freqs(four_idx(fk)), ':', 'Color', [0.5 0.5 0.5], 'Alpha', 0.5);
        end
        xlabel('Frequency (Hz)');
        ylabel('Absorption Coefficient');
        title(sprintf('%s — Spectrum Comparison', target_names{i}));
        legend('Location', 'best');
        grid on;  box on;
        xlim([50, 2000]);
        ylim([0, 1]);

        % 子图2: 误差曲线 (simulated - target)
        subplot(1, 2, 2);
        hold on;
        for m = 1:N_models
            if sim_failed(m, i) == 0
                err = simulated{m, i} - target_spectra(i, :);
                plot(freqs, err, '-', 'Color', colors(m, :), 'LineWidth', 1.2, ...
                    'DisplayName', sprintf('%s (MSE=%.4f)', MODEL_IDS{m}, ...
                    mean(err.^2)));
            end
        end
        yline(0, 'k-');
        xlabel('Frequency (Hz)');
        ylabel('Error (sim - target)');
        title(sprintf('%s — Error Curve', target_names{i}));
        legend('Location', 'best');
        grid on;  box on;
        xlim([50, 2000]);

        % 保存
        plot_fname = fullfile(task_dir, sprintf('%s_ablation_%s.png', DATE, target_names{i}));
        saveas(gcf, plot_fname);
        fprintf('  Plot saved: %s\n', plot_fname);
        close(gcf);
    end

    % --- 模型排名汇总图 ---
    figure('Position', [150, 150, 800, 500]);
    mse_matrix = zeros(N_targets, N_models);
    for i = 1:N_targets
        for m = 1:N_models
            if sim_failed(m, i) == 0
                mse_matrix(i, m) = mean((simulated{m, i} - target_spectra(i, :)).^2);
            else
                mse_matrix(i, m) = NaN;
            end
        end
    end

    mse_matrix(isnan(mse_matrix)) = 0;
    b = bar(mse_matrix);
    for m = 1:length(b)
        if isgraphics(b(m), 'bar')
            b(m).FaceColor = colors(m, :);
        end
    end
    set(gca, 'XTickLabel', target_names);
    xlabel('Target');
    ylabel('MSE');
    title(sprintf('%s — Ablation MSE Comparison', TASK_LABELS{tk}));
    legend(MODEL_IDS, 'Location', 'bestoutside');
    grid on;  box on;

    summary_plot = fullfile(task_dir, sprintf('%s_ablation_summary.png', DATE));
    saveas(gcf, summary_plot);
    fprintf('  Summary plot saved: %s\n', summary_plot);
    close(gcf);

    fprintf('\n=== %s DONE ===\n', TASK_LABELS{tk});
end

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('All ablations verified. Results saved to txt/broadband/ and txt/hologram/\n');
fprintf('%s\n', repmat('=', 1, 70));
