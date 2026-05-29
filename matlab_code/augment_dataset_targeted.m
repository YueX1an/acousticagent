% =========================================================================
% augment_dataset_targeted.m
% Phase 3: Targeted Data Augmentation — based on Phase 2 diagnosis
% =========================================================================
%
% Diagnosis-driven augmentation (from diagnose_params/freqs/bins):
%   P1 [10,000]: 800-820 Hz cluster — data diversity (GA anchor strong, model weak)
%   P0 [ 5,000]: 400 Hz [0.7-1.0] bin — high-absorption sparse (only 3.9% at ≥0.9)
%   P2 [ 5,000]: Lu param variation — hardest param (19.2% relative error)
%
% Total new samples: 20,000 (10% of original 200K)
%
% Method:
%   1. Load existing dataset as seed pool
%   2. Select seeds near target absorption bins
%   3. Perturb with targeted noise (larger on Lu/h for 400Hz, on all for 800Hz)
%   4. Validate all 7 physical constraints
%   5. Run TMM simulation (parfor)
%   6. Concatenate with original dataset, shuffle, save
%
% Usage: Run directly in MATLAB. Ensure helper .m files are on path.
% =========================================================================

clc; clear; close all;

%% ==================== 0. Configuration ====================

% Augmentation sizes
N_P1_800Hz = 10000;   % P1: 800-820 Hz diversity
N_P0_400Hz =  5000;   % P0: 400 Hz high-absorption
N_P2_Lu    =  5000;   % P2: Lu parameter variation
N_aug = N_P1_800Hz + N_P0_400Hz + N_P2_Lu;

% Structural constants (must match generate_strategic_dataset.m)
nlayer = 2;  ncell  = 4;  nw = 2;
delta  = 1.0;  H = 49.7;  W = 49.7;  nh = ncell / nw;
TARGET_W_SUM = W - (nw + 1) * delta;     % 46.7
TARGET_H_SUM = H - (nh + 1) * delta;     % 46.7
FOUR_DELTA    = 4 * delta;               % 4.0
SAFE_MARGIN_TD = 1.5 * delta;            % 1.5
Z_MARGIN       = 3.0;

% Param bounds (must match generate_strategic_dataset.m)
LB = [...
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, ...   % td (1-8)
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, ...   % d  (9-16)
   10.0, 10.0, 10.0, 10.0, ...                            % h  (17-20)
   15.0, 15.0, 15.0, 15.0, ...                            % Lu (21-24)
   57.0, ...                                                % Ltot(25) fixed
   15.0, 15.0, ...                                          % w  (26-27)
    0.0,  0.0,  0.0,  0.0];                               % tu (28-31)

UB = [...
   200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, ... % td
    30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0, ...  % d
    50.0,  50.0,  50.0,  50.0, ...                               % h
    40.0,  40.0,  40.0,  40.0, ...                               % Lu
    57.0, ...                                                     % Ltot
    40.0,  40.0, ...                                              % w
   200.0, 200.0, 200.0, 200.0];                                  % tu

% Simulation config
fmin = 50;  fstep = 10;  fmax = 2000;
f0   = fmin:fstep:fmax;  fL = length(f0);
mnum = 500;  Norder = 9;

% Key frequency indices (50:10:2000)
IDX_400 = 36;   % (400-50)/10 + 1 = 36
IDX_800 = 76;   % (800-50)/10 + 1 = 76
IDX_820 = 78;   % (820-50)/10 + 1 = 78
IDX_390 = 35;   % (390-50)/10 + 1 = 35
IDX_420 = 38;   % (420-50)/10 + 1 = 38

% GA anchor (from generate_strategic_dataset.m)
GA_BEST = [...
    7.4617428,  2.4924932,  1.4229646,  2.8019668,  1.0639180, 14.478907,  1.0000000,  9.3083567, ...
   11.063156,  10.255968,  10.293949,  11.317911,  19.959003,  8.0606058, 12.831908,  4.9389906, ...
   26.619550,  20.080450,  17.394773,  29.305227, ...
   50.000000,  21.568932,  50.000000,  30.359377, ...
   57.000000, ...
   26.315897,  20.384103, ...
    0.011222627, 14.510600, 0.13122252, 15.312971];
GA_BEST = max(LB, min(UB, GA_BEST));  % clip to bounds

% Paths
DATA_DIR   = 'data';
OUT_DIR    = 'data';
SEED_FILE_SL = fullfile(DATA_DIR, 'new_dataset_slN.txt');
SEED_FILE_AL = fullfile(DATA_DIR, 'new_dataset_alphaM.txt');

rng(20260517);  % fixed seed for reproducibility

%% ==================== 1. Load existing dataset as seed pool ====================
fprintf('=========================================================\n');
fprintf('  Phase 3: Targeted Data Augmentation\n');
fprintf('  Loading existing dataset as seed pool...\n');
fprintf('=========================================================\n');

if ~exist(SEED_FILE_SL, 'file') || ~exist(SEED_FILE_AL, 'file')
    error('Seed dataset not found. Run generate_strategic_dataset.m first.');
end

slN_seed = load(SEED_FILE_SL);
alphaM_seed = load(SEED_FILE_AL);  % (196 x N)
alphaM_seed = alphaM_seed';        % (N x 196) for easier indexing
N_seed = size(slN_seed, 1);
fprintf('Loaded %d seed samples.\n', N_seed);

% Compute absorption at key frequencies for seed selection
abs_400 = alphaM_seed(:, IDX_400);
abs_800 = alphaM_seed(:, IDX_800);
mean_abs_500plus = mean(alphaM_seed(:, f0 >= 500), 2);

%% ==================== 2. P1: 800-820 Hz Diversity (10,000) ====================
fprintf('\n=========================================================\n');
fprintf('  P1: 800-820 Hz Cluster Diversity — %d samples\n', N_P1_800Hz);
fprintf('  Diagnosis: GA anchor=0.93 at 800Hz, but model MAE=0.22\n');
fprintf('  Strategy: Seed selection (abs>0.6 at 800Hz) + heavy perturbation\n');
fprintf('=========================================================\n');

% Select seeds: good absorption at 800Hz
seed_mask_p1 = abs_800 > 0.6;
seed_idx_p1 = find(seed_mask_p1);
n_seeds_p1 = length(seed_idx_p1);
fprintf('  Available seeds: %d (abs>0.6 at 800Hz)\n', n_seeds_p1);

if n_seeds_p1 < 100
    warning('  Few seeds available — lowering threshold to 0.5');
    seed_mask_p1 = abs_800 > 0.5;
    seed_idx_p1 = find(seed_mask_p1);
    n_seeds_p1 = length(seed_idx_p1);
    fprintf('  Available seeds (relaxed): %d\n', n_seeds_p1);
end

slN_p1 = zeros(N_P1_800Hz, 31);
tic;
parfor i = 1:N_P1_800Hz
    % Randomly select a seed
    seed_i = seed_idx_p1(randi(n_seeds_p1));
    seed_sl = slN_seed(seed_i, :);

    % Heavy perturbation: sigma = 10-20% of param value
    sigma_ratio = 0.10 + 0.10 * rand();  % 0.10–0.20

    [sl, ok] = perturb_targeted(seed_sl, sigma_ratio, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
        SAFE_MARGIN_TD, Z_MARGIN, delta, ...
        'all');  % perturb all params for diversity
    if ~ok
        sl = seed_sl;  % fallback to seed
    end
    slN_p1(i, :) = sl;
end
elapsed_p1 = toc;
fprintf('  P1 complete: %.1f sec (%.1f samples/sec)\n', elapsed_p1, N_P1_800Hz/elapsed_p1);

%% ==================== 3. P0: 400 Hz High-Absorption (5,000) ====================
fprintf('\n=========================================================\n');
fprintf('  P0: 400 Hz High-Absorption — %d samples\n', N_P0_400Hz);
fprintf('  Diagnosis: [0.9-1.0] bin only 3.9%% coverage at 400Hz\n');
fprintf('  Strategy: Seeds with abs>0.65 at 400Hz, Lu/h-focused perturbation\n');
fprintf('=========================================================\n');

% Select seeds: medium-to-high absorption at 400Hz
seed_mask_p0 = abs_400 > 0.65;
seed_idx_p0 = find(seed_mask_p0);
n_seeds_p0 = length(seed_idx_p0);
fprintf('  Available seeds: %d (abs>0.65 at 400Hz)\n', n_seeds_p0);

if n_seeds_p0 < 100
    warning('  Few seeds available — lowering threshold to 0.55');
    seed_mask_p0 = abs_400 > 0.55;
    seed_idx_p0 = find(seed_mask_p0);
    n_seeds_p0 = length(seed_idx_p0);
    fprintf('  Available seeds (relaxed): %d\n', n_seeds_p0);
end

% Also use GA anchor as additional seed (GA has 0.74 at 400Hz)
seed_pool_p0 = [seed_idx_p0; 0];  % 0 = use GA anchor
n_seeds_p0_total = length(seed_pool_p0);

slN_p0 = zeros(N_P0_400Hz, 31);
tic;
parfor i = 1:N_P0_400Hz
    seed_i = seed_pool_p0(randi(n_seeds_p0_total));
    if seed_i == 0
        seed_sl = GA_BEST;
    else
        seed_sl = slN_seed(seed_i, :);
    end

    % Lu-focused perturbation: larger noise on Lu(21:24) and h(17:20)
    % These control low-frequency absorption
    sigma_ratio = 0.08 + 0.12 * rand();  % 0.08–0.20

    [sl, ok] = perturb_targeted(seed_sl, sigma_ratio, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
        SAFE_MARGIN_TD, Z_MARGIN, delta, ...
        'Lu_h');  % focus on Lu and h params
    if ~ok
        sl = seed_sl;
    end
    slN_p0(i, :) = sl;
end
elapsed_p0 = toc;
fprintf('  P0 complete: %.1f sec (%.1f samples/sec)\n', elapsed_p0, N_P0_400Hz/elapsed_p0);

%% ==================== 4. P2: Lu Parameter Variation (5,000) ====================
fprintf('\n=========================================================\n');
fprintf('  P2: Lu Parameter Variation — %d samples\n', N_P2_Lu);
fprintf('  Diagnosis: Lu hardest param (19.2%% relative error)\n');
fprintf('  Strategy: Systematically vary Lu across full range [15-40mm]\n');
fprintf('=========================================================\n');

slN_p2 = zeros(N_P2_Lu, 31);
tic;
parfor i = 1:N_P2_Lu
    % Randomly pick a base structure from full dataset
    base_i = randi(N_seed);
    base_sl = slN_seed(base_i, :);

    % Force new Lu values across the full range
    % Strategy: with 50% probability, push Lu to extremes
    if rand() < 0.5
        % Extreme Lu: push 1-2 cavity Lu values toward boundaries
        n_push = randi([1, 2]);
        push_idx = randperm(4, n_push);
        for k = push_idx
            if rand() < 0.5
                base_sl(20 + k) = LB(20 + k) + (UB(20 + k) - LB(20 + k)) * 0.05 * rand();  % near LB
            else
                base_sl(20 + k) = UB(20 + k) - (UB(20 + k) - LB(20 + k)) * 0.05 * rand();  % near UB
            end
        end
    else
        % Moderate Lu variation: randomize all 4 Lu values within [15, 40]
        base_sl(21:24) = LB(21:24) + (UB(21:24) - LB(21:24)) .* rand(1, 4);
    end

    % Re-validate and fix dependent params (tu, d, td)
    [sl, ok] = fix_dependent_params(base_sl, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
        SAFE_MARGIN_TD, Z_MARGIN, delta);
    if ~ok
        sl = slN_seed(base_i, :);  % fallback
    end
    slN_p2(i, :) = sl;
end
elapsed_p2 = toc;
fprintf('  P2 complete: %.1f sec (%.1f samples/sec)\n', elapsed_p2, N_P2_Lu/elapsed_p2);

%% ==================== 5. Merge and validate new samples ====================
fprintf('\n=========================================================\n');
fprintf('  Merging augmentation batches...\n');
fprintf('=========================================================\n');

slN_new = [slN_p1; slN_p0; slN_p2];
clear slN_p1 slN_p0 slN_p2;

% Quick constraint check on all new samples
fprintf('  Validating constraints on %d new samples...\n', N_aug);
n_invalid = 0;
for i = 1:N_aug
    if ~check_all_constraints(slN_new(i,:), TARGET_W_SUM, TARGET_H_SUM, ...
            FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
        n_invalid = n_invalid + 1;
    end
end
fprintf('  Invalid samples: %d / %d (%.2f%%)\n', n_invalid, N_aug, 100*n_invalid/N_aug);

%% ==================== 6. TMM Simulation (new samples only) ====================
fprintf('\n=========================================================\n');
fprintf('  TMM Simulation: %d new samples\n', N_aug);
fprintf('  Estimated time: ~%.0f minutes\n', N_aug * 0.5 / 60);
fprintf('=========================================================\n');

alphaM_new = zeros(N_aug, fL);

tic;
parfor n = 1:N_aug
    sl = slN_new(n, :);
    try
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
            f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
        alphaM_new(n, :) = alpha(:)';
    catch ME
        alphaM_new(n, :) = NaN(1, fL);
    end

    if mod(n, 2000) == 0
        fprintf('  Simulation progress: %d / %d (%.1f%%)\n', n, N_aug, 100*n/N_aug);
    end
end
elapsed_sim = toc;
fprintf('  Simulation complete: %.1f min (avg %.2f sec/sample)\n', ...
    elapsed_sim/60, elapsed_sim/N_aug);

%% ==================== 7. Post-process: remove failed simulations ====================
valid_rows = all(isfinite(alphaM_new), 2);
n_failed = sum(~valid_rows);

if n_failed > 0
    fprintf('\n⚠  %d simulation failures removed.\n', n_failed);
    slN_new = slN_new(valid_rows, :);
    alphaM_new = alphaM_new(valid_rows, :);
end
fprintf('  Valid new samples: %d\n', size(slN_new, 1));

%% ==================== 8. Merge with original dataset ====================
fprintf('\n=========================================================\n');
fprintf('  Merging with original dataset...\n');
fprintf('=========================================================\n');

slN_augmented = [slN_seed; slN_new];
alphaM_augmented = [alphaM_seed; alphaM_new];  % both (N x 196)

% Shuffle
shuffle_idx = randperm(size(slN_augmented, 1));
slN_augmented = slN_augmented(shuffle_idx, :);
alphaM_augmented = alphaM_augmented(shuffle_idx, :);

N_final = size(slN_augmented, 1);
fprintf('  Original: %d  |  New: %d  |  Total: %d\n', N_seed, size(slN_new,1), N_final);

%% ==================== 9. Quick quality assessment ====================
fprintf('\n=========================================================\n');
fprintf('  Augmentation Quality Report\n');
fprintf('=========================================================\n');

alphaM_T = alphaM_augmented';  % (196 x N) for freq indexing

% 400 Hz bin coverage (before vs after)
bins_400 = [0:0.1:1.0];
fprintf('\n--- 400 Hz Bin Coverage (Before → After) ---\n');
fprintf('  Bin          Before       After       Change\n');
fprintf('  ' + repmat('-', 1, 48) + '\n');
for b = 1:10
    lo = bins_400(b); hi = bins_400(b+1);
    before_pct = 100 * sum(abs_400 >= lo & abs_400 < hi) / N_seed;
    after_vals = alphaM_T(IDX_400, :);
    after_pct  = 100 * sum(after_vals >= lo & after_vals < hi) / N_final;
    change_str = sprintf('%+.2f%%', after_pct - before_pct);
    fprintf('  [%.1f-%.1f)    %6.2f%%      %6.2f%%     %s\n', ...
        lo, hi, before_pct, after_pct, change_str);
end

% 800 Hz diversity
fprintf('\n--- 800 Hz Absorption Distribution ---\n');
abs_800_before = alphaM_seed(:, IDX_800);
abs_800_after  = alphaM_T(IDX_800, :);
fprintf('  Before: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
    mean(abs_800_before), std(abs_800_before), min(abs_800_before), max(abs_800_before));
fprintf('  After:  mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
    mean(abs_800_after), std(abs_800_after), min(abs_800_after), max(abs_800_after));

% Lu parameter coverage
fprintf('\n--- Lu Parameter Coverage (Before → After) ---\n');
for k = 1:4
    lu_before = slN_seed(:, 20+k);
    lu_after  = slN_augmented(:, 20+k);
    fprintf('  Lu%d: before [%.1f, %.1f] mean=%.1f  →  after [%.1f, %.1f] mean=%.1f\n', ...
        k, min(lu_before), max(lu_before), mean(lu_before), ...
        min(lu_after), max(lu_after), mean(lu_after));
end

%% ==================== 10. Save ====================
fprintf('\n=========================================================\n');
fprintf('  Saving augmented dataset...\n');
fprintf('=========================================================\n');

% Save as transposed alpha (196 x N) for backward compatibility
save(fullfile(OUT_DIR, 'new_dataset_slN_augmented.txt'), 'slN_augmented', '-ascii', '-double');
save(fullfile(OUT_DIR, 'new_dataset_alphaM_augmented.txt'), 'alphaM_T', '-ascii', '-double');

fprintf('✅ Augmented dataset saved!\n');
fprintf('   Params:  data/new_dataset_slN_augmented.txt    (%d x 31)\n', N_final);
fprintf('   Spectra: data/new_dataset_alphaM_augmented.txt (%d x %d)\n', fL, N_final);
fprintf('\n');
fprintf('   Original samples: %d\n', N_seed);
fprintf('   New samples:      %d\n', size(slN_new, 1));
fprintf('   Total:            %d (+%.1f%%)\n', N_final, 100*(N_final-N_seed)/N_seed);
fprintf('\n');
fprintf('   Next: Update create_new_dataset.py to read the augmented files,\n');
fprintf('         then retrain PhyARNet with λ=0.1 on the augmented dataset.\n');
fprintf('=========================================================\n');

%% ==================== 11. Helper Functions ====================

% =========================================================================
% check_all_constraints — identical to original script
% =========================================================================
function ok = check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
    FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    h  = sl(17:20);  w = sl(26:27);  d = sl(9:16);
    Lu = sl(21:24);  tu = sl(28:31);  Ltot = sl(25);  td = sl(1:8);

    if abs(sum(w) - TARGET_W_SUM) > 0.02
        ok = false; return;
    end
    if abs(h(1)+h(2) - TARGET_H_SUM) > 0.02 || abs(h(3)+h(4) - TARGET_H_SUM) > 0.02
        ok = false; return;
    end
    w_for_d = [w(1);w(1);w(2);w(2);w(1);w(1);w(2);w(2)];
    h_for_d = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    d_max_allowed = min(w_for_d, h_for_d) - FOUR_DELTA;
    if any(d(:) > d_max_allowed)
        ok = false; return;
    end
    for k = 1:4
        if Lu(k) + tu(k) + Z_MARGIN > Ltot
            ok = false; return;
        end
    end
    td_h = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    if any(td(:) > td_h - SAFE_MARGIN_TD)
        ok = false; return;
    end
    if any(d(:) ./ h_for_d > 0.65)
        ok = false; return;
    end
    if any(h(:) / Ltot > 0.95)
        ok = false; return;
    end
    ok = true;
end

% =========================================================================
% sample_pair_sum — identical to original
% =========================================================================
function [v1, v2] = sample_pair_sum(lb, ub, target_sum)
    for attempt = 1:200
        v1 = lb + (ub - lb) * rand();
        v2 = target_sum - v1;
        if v2 >= lb && v2 <= ub
            return;
        end
    end
    v1 = target_sum / 2;  v2 = target_sum / 2;
    v1 = max(lb, min(ub, v1));
    v2 = max(lb, min(ub, v2));
end

% =========================================================================
% perturb_targeted — targeted perturbation with focus mode
% mode: 'all' = uniform perturbation on all params
%       'Lu_h' = larger perturbation on Lu(21:24) and h(17:20)
% =========================================================================
function [sl, success] = perturb_targeted(seed, sigma_ratio, ...
    LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta, mode)

    success = true;
    for attempt = 1:300
        sl = seed;

        % Determine per-param-group sigma
        if strcmp(mode, 'Lu_h')
            sigma_luh = sigma_ratio;
            sigma_other = sigma_ratio * 0.3;  % reduced noise on other params
        else
            sigma_luh = sigma_ratio;
            sigma_other = sigma_ratio;
        end

        % --- td (1:8): tube length ---
        for idx = 1:8
            sigma = max(abs(seed(idx)) * sigma_other, 0.3);
            sl(idx) = seed(idx) + sigma * randn();
            sl(idx) = max(LB(idx), min(UB(idx), sl(idx)));
        end

        % --- Lu (21:24): upper cavity length — FOCUS param ---
        for idx = 21:24
            sigma = max(abs(seed(idx)) * sigma_luh, 1.0);
            sl(idx) = seed(idx) + sigma * randn();
            sl(idx) = max(LB(idx), min(UB(idx), sl(idx)));
        end

        % --- Ltot (25): fixed ---
        sl(25) = 57.0;

        % --- h (17:20): cavity height — FOCUS param ---
        % zero-sum perturbation to maintain constraints
        dh12 = seed(17) * sigma_luh * (2*rand() - 1);
        h1_new = seed(17) + dh12;  h2_new = seed(18) - dh12;
        if h1_new >= LB(17) && h1_new <= UB(17) && h2_new >= LB(18) && h2_new <= UB(18)
            sl(17) = h1_new;  sl(18) = h2_new;
        end
        dh34 = seed(19) * sigma_luh * (2*rand() - 1);
        h3_new = seed(19) + dh34;  h4_new = seed(20) - dh34;
        if h3_new >= LB(19) && h3_new <= UB(19) && h4_new >= LB(20) && h4_new <= UB(20)
            sl(19) = h3_new;  sl(20) = h4_new;
        end

        % --- w (26:27): cavity width ---
        dw = seed(26) * sigma_other * (2*rand() - 1);
        w1_new = seed(26) + dw;  w2_new = seed(27) - dw;
        if w1_new >= LB(26) && w1_new <= UB(26) && w2_new >= LB(27) && w2_new <= UB(27)
            sl(26) = w1_new;  sl(27) = w2_new;
        end

        % --- tu (28:31): Z-axis constrained ---
        tu_ok_flag = true;
        for k = 1:4
            sigma = max(abs(seed(27+k)) * sigma_other, 0.3);
            tu_candidate = seed(27+k) + sigma * randn();
            tu_candidate = max(LB(27+k), tu_candidate);
            max_tu = sl(25) - sl(20+k) - Z_MARGIN;
            sl(27+k) = min(tu_candidate, max_tu);
            if sl(27+k) < LB(27+k), tu_ok_flag = false; break; end
        end
        if ~tu_ok_flag, continue; end

        % --- d (9:16): aperture diameter, constrained ---
        w_for_d = [sl(26);sl(26);sl(27);sl(27);sl(26);sl(26);sl(27);sl(27)];
        h_for_d = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        d_limit = min(min(w_for_d, h_for_d) - FOUR_DELTA, h_for_d * 0.65);
        d_ok_flag = true;
        for j = 1:8
            if d_limit(j) < LB(8+j), d_ok_flag = false; break; end
            sigma = max(abs(seed(8+j)) * sigma_other, 0.3);
            d_candidate = seed(8+j) + sigma * randn();
            actual_ub = min(UB(8+j), d_limit(j));
            sl(8+j) = max(LB(8+j), min(actual_ub, d_candidate));
        end
        if ~d_ok_flag, continue; end

        % --- td constraint: td <= h - 1.5*delta ---
        h_rep = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        for j = 1:8
            max_td = h_rep(j) - SAFE_MARGIN_TD;
            if sl(j) > max_td, sl(j) = max(LB(j), max_td); end
        end

        % Full constraint check
        if check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
                FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
            return;
        end
    end
    sl = seed;
    success = false;
end

% =========================================================================
% fix_dependent_params — recompute dependent params (tu, d, td) after
% changing Lu/h/w, maintaining all physical constraints
% =========================================================================
function [sl, ok] = fix_dependent_params(sl, ...
    LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta)

    ok = true;
    Ltot = sl(25);

    % Fix tu based on new Lu values
    for k = 1:4
        max_tu = Ltot - sl(20+k) - Z_MARGIN;
        if max_tu < LB(27+k)
            sl(27+k) = LB(27+k);  % push to min
        else
            actual_ub = min(UB(27+k), max_tu);
            sl(27+k) = LB(27+k) + (actual_ub - LB(27+k)) * rand();
        end
    end

    % Fix d based on new h, w
    w_for_d = [sl(26);sl(26);sl(27);sl(27);sl(26);sl(26);sl(27);sl(27)];
    h_for_d = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
    d_limit = min(min(w_for_d, h_for_d) - FOUR_DELTA, h_for_d * 0.65);

    for j = 1:8
        if d_limit(j) < LB(8+j)
            ok = false; return;
        end
        actual_ub = min(UB(8+j), d_limit(j));
        sl(8+j) = LB(8+j) + (actual_ub - LB(8+j)) * rand();
    end

    % Fix td based on h
    h_rep = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
    for j = 1:8
        max_td = h_rep(j) - SAFE_MARGIN_TD;
        if max_td < LB(j)
            ok = false; return;
        end
        actual_ub = min(UB(j), max_td);
        sl(j) = LB(j) + (actual_ub - LB(j)) * rand();
    end

    if ~check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
            FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
        ok = false;
    end
end
