% verify_broadband.m
% TMM verification for broadband absorber design (400-2000Hz α≥0.9)
clc; clear; close all;

MAT_FILE = '..\eval_results\design_broadband.mat';
if ~exist(MAT_FILE,'file'), error('Run design_broadband_target.py first.'); end

data = load(MAT_FILE);
freqs = data.freqs(:)';
fL = length(freqs);
target = data.target_broadband(:)';

% TMM config
nlayer=2; ncell=4; nw=2; delta=1.0; mnum=500; Norder=9;

fprintf('========== Broadband Absorber TMM Verification ==========\n');
fprintf('Target: α≥0.9 at 400-2000 Hz, average = %.4f\n', mean(target(freqs>=400)));

model_names = {'Phase1 (λ=0.1)', 'NoSE (λ=0.1, best)'};

fprintf('\n%-20s %12s %12s %12s %12s\n', 'Model', 'Mean α(400+)', 'Min α(400+)', 'Overall MAE', '400Hz α');
fprintf('%s\n', repmat('-',1,70));

for m = 1:2
    if m==1, params = data.params_Phase1_lambda0_1;
    else,    params = data.params_NoSE_lambda0_1; end

    [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
        freqs, params, ncell, nw, delta, nlayer, mnum, Norder);
    alpha = alpha(:)';

    alpha_400plus = alpha(freqs>=400);
    mean_a = mean(alpha_400plus);
    min_a  = min(alpha_400plus);
    mae    = mean(abs(alpha - target));
    a400   = alpha(freqs==400);

    fprintf('%-20s %12.4f %12.4f %12.6f %12.4f\n', model_names{m}, mean_a, min_a, mae, a400);
end

fprintf('\nNext: plot absorption curves vs target.\n');
