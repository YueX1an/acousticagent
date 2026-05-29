% verify_mnist_1907.m
% TMM verification for MNIST "1907" 16×16 hologram
clc; clear; close all;

MAT_FILE = '..\eval_results\design_mnist_1907.mat';
if ~exist(MAT_FILE,'file'), error('Run design_mnist_hologram.py first.'); end

data = load(MAT_FILE);
freqs = data.freqs(:)';
fL = length(freqs);
pattern = double(data.pattern);
H = double(data.grid_shape(1)); W = double(data.grid_shape(2));
N = H * W;

% TMM config
nlayer=2; ncell=4; nw=2; delta=1.0; mnum=500; Norder=9;

four_idx = [35, 85, 135, 185] + 1;  % MATLAB 1-based
four_freq = freqs(four_idx);

alpha_target = data.alpha_target;
alpha_bg     = data.alpha_bg;

fprintf('========== MNIST "1907" Hologram TMM Verification ==========\n');
fprintf('Grid: %d×%d = %d pixels\n', H, W, N);
fprintf('Target α=%.1f, Background α=%.1f at [%d %d %d %d] Hz\n', ...
    alpha_target, alpha_bg, four_freq);
fprintf('Total TMM simulations: %d\n', N*2);

% ---- Run TMM on all pixels (both models) ----
model_names = {'Phase1', 'NoSE'};
alpha_all = cell(2, 1);  % each: (fL, N)

for m = 1:2
    if m==1, params_all = data.params_Phase1_lambda0_1;
    else,    params_all = data.params_NoSE_lambda0_1; end

    alpha_m = zeros(fL, N);
    fprintf('\nSimulating %s...\n', model_names{m});

    parfor i = 1:N
        try
            [a, ~] = fun_structure_broadband_nlayer_HB_func(...
                freqs, params_all(i,:), ncell, nw, delta, nlayer, mnum, Norder);
            alpha_m(:, i) = a(:);
        catch
            alpha_m(:, i) = NaN(fL, 1);
        end
    end
    alpha_all{m} = alpha_m;
end

n_fail = sum(~all(isfinite(alpha_all{1}),1)) + sum(~all(isfinite(alpha_all{2}),1));
fprintf('\nSimulation failures: %d / %d\n', n_fail, N*2);

% ---- Per-pixel 4-frequency accuracy ----
fprintf('\n========== 4-Frequency Accuracy ==========\n');
for m = 1:2
    fprintf('\n%s:\n', model_names{m});
    alpha_m = alpha_all{m};

    for fi = 1:4
        idx_f = four_idx(fi);
        pixel_vals = alpha_m(idx_f, :);  % (1, N)
        target_vals = alpha_target * pattern(:)' + alpha_bg * (1 - pattern(:)');

        mae_f = mean(abs(pixel_vals - target_vals));
        target_px = pixel_vals(pattern(:)==1);
        bg_px     = pixel_vals(pattern(:)==0);
        contrast = mean(target_px) - mean(bg_px);

        fprintf('  %d Hz: MAE=%.4f  Target=%.3f±%.3f  BG=%.3f±%.3f  Contrast=%.3f\n', ...
            four_freq(fi), mae_f, mean(target_px), std(target_px), ...
            mean(bg_px), std(bg_px), contrast);
    end
end

% ---- Save results ----
save('..\eval_results\verify_mnist_1907_results.mat', ...
    'alpha_all', 'model_names', 'pattern', 'freqs', 'four_idx', 'four_freq');
fprintf('\nSaved: eval_results/verify_mnist_1907_results.mat\n');
fprintf('========== Done ==========\n');
