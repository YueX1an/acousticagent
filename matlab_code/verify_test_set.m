% verify_test_set.m
% Extended closed-loop verification: independent TMM validation
% Compares λ=0.0 (T00) vs λ=0.1 (Phase1) on 100 random test samples
% Run validate_test_set.py first to generate test_set_verify.mat
clc; clear; close all;

%% ==================== Configuration ====================
EVAL_DIR = '..\eval_results';
MAT_FILE = fullfile(EVAL_DIR, 'test_set_verify.mat');

% TMM simulation params
nlayer = 2;  ncell = 4;  nw = 2;  delta = 1.0;
mnum = 500;  Norder = 9;

%% ==================== Load data ====================
fprintf('========== Extended Closed-Loop Verification ==========\n');
fprintf('Loading %s...\n', MAT_FILE);

if ~exist(MAT_FILE, 'file')
    error('File not found: %s.\nRun validate_test_set.py first.', MAT_FILE);
end

data = load(MAT_FILE);
target_spectra = data.target_spectra;     % (N, 196) ground truth
freqs = data.freqs(:)';                   % ensure row vector (1×196) for TMM function
fL = length(freqs);
N = size(target_spectra, 1);

% Detect model param fields
model_fields = fieldnames(data);
model_fields = model_fields(startsWith(model_fields, 'params_'));
N_models = length(model_fields);

model_labels = strings(N_models, 1);
model_short  = strings(N_models, 1);
for j = 1:N_models
    model_short(j) = erase(model_fields{j}, 'params_');
    if contains(model_short(j), 'T00')
        model_labels(j) = '\lambda=0.0 (T00)';
    elseif contains(model_short(j), 'Phase1')
        model_labels(j) = '\lambda=0.1 (Phase1)';
    else
        model_labels(j) = model_short(j);
    end
end

fprintf('Loaded: %d samples, %d freqs (%d-%d Hz)\n', N, fL, freqs(1), freqs(end));
fprintf('Models to verify: %d\n', N_models);
for j = 1:N_models
    fprintf('  [%d] %s\n', j, model_labels(j));
end
fprintf('Total TMM simulations: %d\n\n', N * N_models);

%% ==================== Run TMM simulations ====================
% Build param matrix: (N * N_models) x 31
slN_verify = zeros(N * N_models, 31);
sample_map = zeros(N * N_models, 2);  % (sample_idx, model_idx)

row = 1;
for m = 1:N_models
    params_m = data.(model_fields{m});  % (N, 31)
    for i = 1:N
        slN_verify(row, :) = params_m(i, :);
        sample_map(row, :) = [i, m];
        row = row + 1;
    end
end

N_total = size(slN_verify, 1);

fprintf('========== Running TMM Simulations ==========\n');
fprintf('Total: %d simulations\n', N_total);
tic;

alphaM_sim = zeros(fL, N_total);

parfor row = 1:N_total
    sl = slN_verify(row, :);
    try
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
            freqs, sl, ncell, nw, delta, nlayer, mnum, Norder);
        alphaM_sim(:, row) = alpha(:);
    catch ME
        alphaM_sim(:, row) = NaN(fL, 1);
        fprintf('  ⚠ Sample %d failed: %s\n', row, ME.message);
    end

    if mod(row, 50) == 0
        fprintf('  Progress: %d / %d (%.0f%%)\n', row, N_total, 100*row/N_total);
    end
end

elapsed = toc;
fprintf('Done. Elapsed: %.1f min (avg %.2f sec/sample)\n', elapsed/60, elapsed/N_total);

% Check for failures
n_failed = sum(~all(isfinite(alphaM_sim), 1));
if n_failed > 0
    fprintf('⚠  %d simulations failed (NaN/Inf).\n', n_failed);
end

%% ==================== Per-sample analysis ====================
fprintf('\n========== Per-Sample Analysis ==========\n');

% Organize results by model
results = struct();
for m = 1:N_models
    rows_m = find(sample_map(:, 2) == m);
    alpha_m = alphaM_sim(:, rows_m);  % (196, N)

    % Per-sample MAE
    mae_per_sample = zeros(N, 1);
    for i = 1:N
        true_spec = target_spectra(i, :)';
        mae_per_sample(i) = mean(abs(alpha_m(:, i) - true_spec));
    end

    results(m).name  = model_labels(m);
    results(m).short = model_short(m);
    results(m).mae   = mae_per_sample;
    results(m).mean_mae = mean(mae_per_sample);
    results(m).std_mae  = std(mae_per_sample);
    results(m).median_mae = median(mae_per_sample);
    results(m).min_mae = min(mae_per_sample);
    results(m).max_mae = max(mae_per_sample);

    fprintf('\n  %s:\n', model_labels(m));
    fprintf('    Mean MAE:   %.6f\n', results(m).mean_mae);
    fprintf('    Median MAE: %.6f\n', results(m).median_mae);
    fprintf('    Std MAE:    %.6f\n', results(m).std_mae);
    fprintf('    Range:      [%.6f, %.6f]\n', results(m).min_mae, results(m).max_mae);
end

%% ==================== Head-to-head comparison ====================
fprintf('\n========== Head-to-Head Comparison ==========\n');

% Per-sample winner
wins_m1 = 0; wins_m2 = 0; ties = 0;
diffs = results(2).mae - results(1).mae;  % λ=0.1 - λ=0.0: negative = λ=0.1 better

for i = 1:N
    if diffs(i) < -0.0001
        wins_m2 = wins_m2 + 1;
    elseif diffs(i) > 0.0001
        wins_m1 = wins_m1 + 1;
    else
        ties = ties + 1;
    end
end

fprintf('\n  Per-sample winner count:\n');
fprintf('    λ=0.0 wins: %d / %d (%.0f%%)\n', wins_m1, N, 100*wins_m1/N);
fprintf('    λ=0.1 wins: %d / %d (%.0f%%)\n', wins_m2, N, 100*wins_m2/N);
fprintf('    Ties:       %d\n', ties);

mean_diff = mean(diffs);
fprintf('\n  Mean MAE difference (λ=0.1 - λ=0.0): %+.6f\n', mean_diff);
if mean_diff < 0
    fprintf('  → λ=0.1 is BETTER by %.2f%% on real TMM simulation\n', ...
        -100 * mean_diff / results(1).mean_mae);
else
    fprintf('  → λ=0.0 is BETTER by %.2f%% on real TMM simulation\n', ...
        100 * mean_diff / results(2).mean_mae);
end

%% ==================== By frequency band ====================
fprintf('\n========== Per-Band Error ==========\n');

band_names  = {'50-400Hz', '410-900Hz', '910-1400Hz', '1410-2000Hz'};
band_ranges = {1:36, 37:86, 87:136, 137:196};

fprintf('  %-14s  %-16s  %-16s  %-12s\n', 'Band', model_labels(1), model_labels(2), 'Δ (λ0.1-λ0.0)');
fprintf('  %s\n', repmat('-', 1, 65));

for b = 1:4
    idx = band_ranges{b};

    mae_b_m1 = zeros(N, 1);
    mae_b_m2 = zeros(N, 1);
    for i = 1:N
        true_spec = target_spectra(i, idx)';
        mae_b_m1(i) = mean(abs(alphaM_sim(idx, i) - true_spec));           % model1 col i
        mae_b_m2(i) = mean(abs(alphaM_sim(idx, N + i) - true_spec));       % model2 col N+i
    end

    m1_mean = mean(mae_b_m1);
    m2_mean = mean(mae_b_m2);
    delta = m2_mean - m1_mean;

    fprintf('  %-14s  %-16.6f  %-16.6f  %+12.6f\n', band_names{b}, m1_mean, m2_mean, delta);
end

%% ==================== Summary verdict ====================
fprintf('\n========== VERDICT ==========\n');

if mean_diff < 0
    fprintf('✅ λ=0.1 (physics-informed) outperforms λ=0.0 on independent TMM validation.\n');
    fprintf('   Improvement: %.2f%% (mean MAE across %d random test samples)\n', ...
        -100 * mean_diff / results(1).mean_mae, N);
    fprintf('   Win rate: %d/%d (%.0f%%)\n', wins_m2, N, 100*wins_m2/N);
elseif mean_diff > 0
    fprintf('❌ λ=0.0 (pure data) outperforms λ=0.1 on independent TMM validation.\n');
    fprintf('   λ=0.0 is better by %.2f%%\n', 100 * mean_diff / results(2).mean_mae);
    fprintf('   Win rate for λ=0.0: %d/%d (%.0f%%)\n', wins_m1, N, 100*wins_m1/N);
else
    fprintf('➡  λ=0.0 and λ=0.1 show no significant difference on TMM.\n');
end

fprintf('\n  Robustness test (ResNetDNN proxy): claimed -59%% for λ=0.1\n');
fprintf('  This test (MATLAB TMM):            %.1f%% for λ=0.1\n', ...
    -100 * mean_diff / max(results(1).mean_mae, 1e-8));
fprintf('  → The proxy-based test may over/under-estimate the true gap.\n');

%% ==================== Save results ====================
save_path = fullfile(EVAL_DIR, 'test_set_verify_results.mat');
save(save_path, 'results', 'diffs', 'wins_m1', 'wins_m2', 'ties', ...
    'mean_diff', 'alphaM_sim', 'sample_map', 'target_spectra', 'freqs', ...
    'model_labels', 'model_short');

fprintf('\nResults saved to: %s\n', save_path);
fprintf('========== Verification Complete ==========\n');
