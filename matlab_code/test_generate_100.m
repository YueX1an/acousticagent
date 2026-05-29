% =========================================================================
% test_generate_100.m
% 小型验证脚本 — 生成 100 组数据, 逐项检查数据生成管线是否正常
% =========================================================================
clc; clear; close all;

%% --- 配置 (与 generate_strategic_dataset.m 完全一致) ---
PART_SIZES = [40, 30, 20, 10];   % A/B/C/D 各40/30/20/10, 共100
N_total = sum(PART_SIZES);

nlayer = 2;  ncell = 4;  nw = 2;  delta = 1.0;
H = 49.7;  W = 49.7;  nh = ncell / nw;
TARGET_W_SUM = W - (nw + 1) * delta;
TARGET_H_SUM = H - (nh + 1) * delta;
FOUR_DELTA = 4 * delta;
SAFE_MARGIN_TD = 1.5 * delta;
Z_MARGIN = 3.0;

LB = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, ...
      1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, ...
      10.0,10.0,10.0,10.0, ...
      15.0,15.0,15.0,15.0, ...
      57.0, ...
      15.0,15.0, ...
      0.0,0.0,0.0,0.0];
UB = [200.0,200.0,200.0,200.0,200.0,200.0,200.0,200.0, ...
      30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0, ...
      50.0,50.0,50.0,50.0, ...
      40.0,40.0,40.0,40.0, ...
      57.0, ...
      40.0,40.0, ...
      200.0,200.0,200.0,200.0];

GA_BEST = [7.4617428,2.4924932,1.4229646,2.8019668,1.0639180,14.478907,1.0000000,9.3083567, ...
           11.063156,10.255968,10.293949,11.317911,19.959003,8.0606058,12.831908,4.9389906, ...
           26.619550,20.080450,17.394773,29.305227, ...
           50.000000,21.568932,50.000000,30.359377, ...
           57.000000, ...
           26.315897,20.384103, ...
           0.011222627,14.510600,0.13122252,15.312971];
GA_BEST_CLIPPED = max(LB, min(UB, GA_BEST));
GA_Lu_original = GA_BEST(21:24);

fmin = 50;  fstep = 10;  fmax = 2000;
f0 = fmin:fstep:fmax;
fL = length(f0);
mnum = 500;  Norder = 9;

rng(42);

%% ==================== 1. 生成 100 组参数 ====================
fprintf('========== 生成 100 组参数 ==========\n');

% Part A: 随机 (40)
fprintf('Part A (40)...');
slN_A = zeros(PART_SIZES(1), 31);
for i = 1:PART_SIZES(1)
    [sl, ~] = generate_one_random(LB, UB, TARGET_W_SUM, TARGET_H_SUM, ...
        FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta);
    slN_A(i, :) = sl;
end
fprintf(' done.\n');

% Part B: GA微扰 (30)
fprintf('Part B (30)...');
slN_B = zeros(PART_SIZES(2), 31);
noise_levels = [0.03, 0.08, 0.15];
noise_weights = [0.50, 0.30, 0.20];
noise_cumsum = cumsum(noise_weights);
for i = 1:PART_SIZES(2)
    r = rand();
    if r < noise_cumsum(1),      sigma_ratio = noise_levels(1);
    elseif r < noise_cumsum(2),  sigma_ratio = noise_levels(2);
    else,                        sigma_ratio = noise_levels(3);
    end
    [sl, ~] = perturb_around_anchor(GA_BEST_CLIPPED, GA_Lu_original, sigma_ratio, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
        Z_MARGIN, delta);
    slN_B(i, :) = sl;
end
fprintf(' done.\n');

% Part C: 分层 (20)
fprintf('Part C (20)...');
slN_C = zeros(PART_SIZES(3), 31);
prof_h_range = {[30,50],[10,25],[30,50],[10,50],[10,50]};
prof_w_range = {[25,40],[25,40],[15,25],[15,40],[15,40]};
samples_per_prof = floor(PART_SIZES(3) / 5);
prof_counts = samples_per_prof * ones(1, 5);
prof_counts(1) = prof_counts(1) + mod(PART_SIZES(3), 5);
prof_start_idx = [1, cumsum(prof_counts(1:end-1)) + 1];
for prof = 1:5
    n_prof = prof_counts(prof);
    start_i = prof_start_idx(prof);
    h_lo = prof_h_range{prof}(1);  h_hi = prof_h_range{prof}(2);
    w_lo = prof_w_range{prof}(1);  w_hi = prof_w_range{prof}(2);
    is_diff_h = (prof == 5);
    for j = 1:n_prof
        [sl, ~] = generate_one_stratified(h_lo, h_hi, w_lo, w_hi, is_diff_h, ...
            LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
            Z_MARGIN, delta);
        slN_C(start_i + j - 1, :) = sl;
    end
end
fprintf(' done.\n');

% Part D: 边界 (10)
fprintf('Part D (10)...');
slN_D = zeros(PART_SIZES(4), 31);
for i = 1:PART_SIZES(4)
    strategy = mod(i - 1, 7) + 1;
    [sl, ~] = generate_one_boundary(strategy, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
        Z_MARGIN, delta);
    slN_D(i, :) = sl;
end
fprintf(' done.\n');

% 合并
slN = [slN_A; slN_B; slN_C; slN_D];
fprintf('\n合并样本数: %d\n', size(slN, 1));

%% ==================== 2. 逐条验证所有约束 ====================
fprintf('\n========== 逐条约束验证 ==========\n');
n = size(slN, 1);
violations = zeros(7, 1);

for i = 1:n
    sl = slN(i, :)';
    h = sl(17:20);  w = sl(26:27);  d = sl(9:16);
    Lu = sl(21:24);  tu = sl(28:31);  Ltot = sl(25);  td = sl(1:8);

    % 约束1: w1 + w2 = 46.7
    if abs(sum(w) - TARGET_W_SUM) > 0.02, violations(1) = violations(1) + 1; end

    % 约束2: h1+h2 = h3+h4 = 46.7
    if abs(h(1)+h(2)-TARGET_H_SUM)>0.02 || abs(h(3)+h(4)-TARGET_H_SUM)>0.02
        violations(2) = violations(2) + 1;
    end

    % 约束3: d <= min(w,h) - 4δ
    w_for_d = [w(1);w(1);w(2);w(2);w(1);w(1);w(2);w(2)];
    h_for_d = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    if any(d > min(w_for_d, h_for_d) - FOUR_DELTA), violations(3) = violations(3) + 1; end

    % 约束4: Lu + tu + 3 <= Ltot
    if any(Lu + tu + Z_MARGIN > Ltot), violations(4) = violations(4) + 1; end

    % 约束5: td <= h - 1.5δ
    td_h = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    if any(td > td_h - SAFE_MARGIN_TD), violations(5) = violations(5) + 1; end

    % 约束6: d / h <= 0.65
    if any(d ./ h_for_d > 0.65), violations(6) = violations(6) + 1; end

    % 约束7: h / Ltot <= 0.95
    if any(h / Ltot > 0.95), violations(7) = violations(7) + 1; end
end

constraint_names = {'w1+w2=46.7','h1+h2=h3+h4=46.7','d<=min(w,h)-4δ','Lu+tu+3<=Ltot','td<=h-1.5δ','d/h<=0.65','h/Ltot<=0.95'};
all_pass = true;
for c = 1:7
    if violations(c) == 0
        fprintf('  ✅ 约束%d (%s): 0 违规\n', c, constraint_names{c});
    else
        fprintf('  ❌ 约束%d (%s): %d 违规!!!\n', c, constraint_names{c}, violations(c));
        all_pass = false;
    end
end

% 边界检查
fprintf('\n--- 全局边界检查 ---\n');
below_lb = sum(slN < LB, 1);
above_ub = sum(slN > UB, 1);
if any(below_lb > 0)
    fprintf('  ❌ %d 个参数出现过低于 LB\n', sum(below_lb > 0));
    all_pass = false;
else
    fprintf('  ✅ 所有参数 >= LB\n');
end
if any(above_ub > 0)
    fprintf('  ❌ %d 个参数出现过高于 UB\n', sum(above_ub > 0));
    all_pass = false;
else
    fprintf('  ✅ 所有参数 <= UB\n');
end

%% ==================== 3. 参数分布快速摘要 ====================
fprintf('\n========== 参数范围摘要 ==========\n');
param_names = {'td1','td2','td3','td4','td5','td6','td7','td8', ...
               'd1','d2','d3','d4','d5','d6','d7','d8', ...
               'h1','h2','h3','h4','Lu1','Lu2','Lu3','Lu4', ...
               'Ltot','w1','w2','tu1','tu2','tu3','tu4'};
for k = 1:31
    v = slN(:, k);
    fprintf('  %s: [%.2f, %.2f]  (允许 [%.0f, %.0f])\n', ...
        param_names{k}, min(v), max(v), LB(k), UB(k));
end

%% ==================== 4. 仿真测试 (仅前20组, 节省时间) ====================
fprintf('\n========== 仿真测试 (前20组) ==========\n');
N_sim = min(20, size(slN, 1));
alphaM = zeros(fL, N_sim);
sim_fail = 0;

for i = 1:N_sim
    try
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
            f0, slN(i, :), ncell, nw, delta, nlayer, mnum, Norder);
        alphaM(:, i) = alpha(:);
        if mod(i, 5) == 0, fprintf('  %d/%d...\n', i, N_sim); end
    catch ME
        fprintf('  ❌ 样本 %d 仿真失败: %s\n', i, ME.message);
        alphaM(:, i) = NaN(fL, 1);
        sim_fail = sim_fail + 1;
    end
end

if sim_fail == 0
    fprintf('  ✅ 全部 %d 组仿真成功\n', N_sim);
else
    fprintf('  ⚠️ %d 组仿真失败\n', sim_fail);
    all_pass = false;
end

%% ==================== 5. 吸收曲线统计 ====================
fprintf('\n--- 吸收曲线快速统计 (仿真成功的样本) ---\n');
valid = all(isfinite(alphaM), 1);
if any(valid)
    valid_alpha = alphaM(:, valid);
    mean_abs = mean(valid_alpha, 1);
    mean_500plus = mean(valid_alpha(f0 >= 500, :), 1);
    fprintf('  全频段均值: %.4f ± %.4f  [%.4f, %.4f]\n', ...
        mean(mean_abs), std(mean_abs), min(mean_abs), max(mean_abs));
    fprintf('  500Hz+ 均值: %.4f ± %.4f  [%.4f, %.4f]\n', ...
        mean(mean_500plus), std(mean_500plus), min(mean_500plus), max(mean_500plus));

    % 四频点分布
    target_freqs = [400, 900, 1400, 1900];
    target_idx = round((target_freqs - fmin) / fstep) + 1;
    fprintf('  四频点吸收范围:\n');
    for fi = 1:4
        v = valid_alpha(target_idx(fi), :);
        fprintf('    %d Hz: [%.3f, %.3f]  mean=%.3f\n', target_freqs(fi), min(v), max(v), mean(v));
    end
end

%% ==================== 6. 最终结论 ====================
fprintf('\n========== 最终结论 ==========\n');
if all_pass
    fprintf('✅ 所有检查通过! 数据生成代码工作正常。\n');
else
    fprintf('❌ 存在问题, 请检查上述标记为 ❌ 的项。\n');
end

% =========================================================================
% 局部函数 (从 generate_strategic_dataset.m 复制, 完全一致)
% =========================================================================

function [v1, v2] = sample_pair_sum(lb, ub, target_sum)
    for attempt = 1:200
        v1 = lb + (ub - lb) * rand();
        v2 = target_sum - v1;
        if v2 >= lb && v2 <= ub, return; end
    end
    v1 = target_sum / 2;
    v2 = target_sum / 2;
    v1 = max(lb, min(ub, v1));
    v2 = max(lb, min(ub, v2));
end

function [v1, v2] = sample_pair_sum_stratified(lb_primary, ub_primary, lb_other, ub_other, target_sum)
% 分层采样: v1 在 [lb_primary, ub_primary] 内, v2 只需在 [lb_other, ub_other] 内
    for attempt = 1:200
        v1 = lb_primary + (ub_primary - lb_primary) * rand();
        v2 = target_sum - v1;
        if v2 >= lb_other && v2 <= ub_other, return; end
    end
    v1 = (lb_primary + ub_primary) / 2;
    v2 = target_sum - v1;
    v1 = max(lb_primary, min(ub_primary, v1));
    v2 = max(lb_other, min(ub_other, v2));
end

function ok = check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
    FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    h  = sl(17:20);  w  = sl(26:27);  d  = sl(9:16);
    Lu = sl(21:24);  tu = sl(28:31);  Ltot = sl(25);  td = sl(1:8);
    if abs(sum(w) - TARGET_W_SUM) > 0.02, ok = false; return; end
    if abs(h(1)+h(2)-TARGET_H_SUM)>0.02 || abs(h(3)+h(4)-TARGET_H_SUM)>0.02
        ok = false; return;
    end
    w_for_d = [w(1);w(1);w(2);w(2);w(1);w(1);w(2);w(2)];
    h_for_d = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    if any(d(:) > min(w_for_d, h_for_d) - FOUR_DELTA), ok = false; return; end
    for k = 1:4
        if Lu(k)+tu(k)+Z_MARGIN > Ltot, ok = false; return; end
    end
    td_h = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    if any(td(:)>td_h-SAFE_MARGIN_TD), ok = false; return; end
    if any(d(:)./h_for_d>0.65), ok = false; return; end
    if any(h(:)/Ltot>0.95), ok = false; return; end
    ok = true;
end

function [d_vec, success] = generate_safe_d(w, h, LB_d, UB_d, FOUR_DELTA)
    w_for_d = [w(1);w(1);w(2);w(2);w(1);w(1);w(2);w(2)];
    h_for_d = [h(1);h(2);h(3);h(4);h(1);h(2);h(3);h(4)];
    d_limit = min(min(w_for_d, h_for_d) - FOUR_DELTA, h_for_d * 0.65);
    if any(d_limit < LB_d), d_vec = zeros(8,1); success = false; return; end
    d_vec = zeros(8,1);
    for i = 1:8
        actual_ub = min(UB_d(i), d_limit(i));
        d_vec(i) = LB_d(i) + (actual_ub - LB_d(i)) * rand();
    end
    success = true;
end

function [tu_vec, success] = generate_safe_tu(Lu, Ltot, LB_tu, UB_tu, Z_MARGIN)
    tu_vec = zeros(4,1);
    for k = 1:4
        max_tu = Ltot - Lu(k) - Z_MARGIN;
        if max_tu < LB_tu(k), success = false; return; end
        actual_ub = min(UB_tu(k), max_tu);
        tu_vec(k) = LB_tu(k) + (actual_ub - LB_tu(k)) * rand();
    end
    success = true;
end

function [sl, rejections] = generate_one_random(LB, UB, TARGET_W_SUM, ...
    TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    rejections = 0;
    for attempt = 1:500
        sl = zeros(1,31);
        % 先生成独立参数和耦合参数, 再生成依赖参数
        sl(21:24) = LB(21:24) + (UB(21:24)-LB(21:24)).*rand(1,4);  % Lu
        sl(25) = 57.0;                                                % Ltot
        [sl(17),sl(18)] = sample_pair_sum(LB(17),UB(17),TARGET_H_SUM); % h1,h2
        [sl(19),sl(20)] = sample_pair_sum(LB(19),UB(19),TARGET_H_SUM); % h3,h4
        [sl(26),sl(27)] = sample_pair_sum(LB(26),UB(26),TARGET_W_SUM); % w1,w2
        % td: 在 h 约束下采样 (td(i) <= h_row_i - 1.5*delta)
        h_for_td = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections=rejections+1; continue; end
        % tu: Z轴安全
        [tu_vec,tu_ok] = generate_safe_tu(sl(21:24),sl(25),LB(28:31),UB(28:31),Z_MARGIN);
        if ~tu_ok, rejections=rejections+1; continue; end
        sl(28:31)=tu_vec;
        % d: 物理安全
        [d_vec,d_ok]=generate_safe_d(sl(26:27),sl(17:20),LB(9:16),UB(9:16),FOUR_DELTA);
        if ~d_ok, rejections=rejections+1; continue; end
        sl(9:16)=d_vec;
        if check_all_constraints(sl,TARGET_W_SUM,TARGET_H_SUM,FOUR_DELTA,SAFE_MARGIN_TD,Z_MARGIN,delta)
            return;
        end
        rejections=rejections+1;
    end
    error('generate_one_random: 无法生成合法样本。');
end

function [sl, success] = perturb_around_anchor(anchor, Lu_original, ...
    sigma_ratio, LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta)
    success = true;
    for attempt = 1:500
        sl = anchor;
        for idx = [1:8, 21:24]
            sigma = max(abs(anchor(idx))*sigma_ratio, 0.3);
            sl(idx) = anchor(idx) + sigma*randn();
            sl(idx) = max(LB(idx), min(UB(idx), sl(idx)));
        end
        dh12 = anchor(17)*sigma_ratio*(2*rand()-1);
        h1_new=anchor(17)+dh12; h2_new=anchor(18)-dh12;
        if h1_new>=LB(17)&&h1_new<=UB(17)&&h2_new>=LB(18)&&h2_new<=UB(18)
            sl(17)=h1_new; sl(18)=h2_new;
        end
        dh34 = anchor(19)*sigma_ratio*(2*rand()-1);
        h3_new=anchor(19)+dh34; h4_new=anchor(20)-dh34;
        if h3_new>=LB(19)&&h3_new<=UB(19)&&h4_new>=LB(20)&&h4_new<=UB(20)
            sl(19)=h3_new; sl(20)=h4_new;
        end
        dw = anchor(26)*sigma_ratio*(2*rand()-1);
        w1_new=anchor(26)+dw; w2_new=anchor(27)-dw;
        if w1_new>=LB(26)&&w1_new<=UB(26)&&w2_new>=LB(27)&&w2_new<=UB(27)
            sl(26)=w1_new; sl(27)=w2_new;
        end
        tu_ok = true;
        for k = 1:4
            sigma = max(abs(anchor(27+k))*sigma_ratio, 0.3);
            tu_candidate = anchor(27+k) + sigma*randn();
            tu_candidate = max(LB(27+k), tu_candidate);
            max_tu = sl(25) - sl(20+k) - Z_MARGIN;
            sl(27+k) = min(tu_candidate, max_tu);
            if sl(27+k)<LB(27+k), tu_ok=false; break; end
        end
        if ~tu_ok, continue; end
        [d_vec, d_ok] = generate_safe_d(sl(26:27), sl(17:20), LB(9:16), UB(9:16), FOUR_DELTA);
        if ~d_ok, continue; end
        for j = 1:8
            sigma = max(abs(anchor(8+j))*sigma_ratio, 0.3);
            d_candidate = anchor(8+j) + sigma*randn();
            d_candidate = max(LB(8+j), min(d_vec(j), d_candidate));
            if d_candidate<=d_vec(j), sl(8+j)=d_candidate;
            else, sl(8+j)=d_vec(j); end
        end
        h_rep = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        for j=1:8
            max_td = h_rep(j)-SAFE_MARGIN_TD;
            if sl(j)>max_td, sl(j)=max(LB(j),max_td); end
        end
        if check_all_constraints(sl,TARGET_W_SUM,TARGET_H_SUM,FOUR_DELTA,SAFE_MARGIN_TD,Z_MARGIN,delta)
            return;
        end
    end
    sl = anchor;
    success = false;
end

function [sl, rejections] = generate_one_stratified(h_lo, h_hi, w_lo, w_hi, ...
    is_diff_h, LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta)
    rejections = 0;
    for attempt = 1:500
        sl = zeros(1,31);
        sl(21:24) = LB(21:24)+(UB(21:24)-LB(21:24)).*rand(1,4);  % Lu
        sl(25) = 57.0;                                              % Ltot
        if is_diff_h
            for attempt_h = 1:50
                h1 = h_lo+(h_hi-h_lo)*rand();
                h2 = TARGET_H_SUM-h1;
                if h2>=LB(17)&&h2<=UB(17)&&abs(h1-h2)>=10
                    sl(17)=h1; sl(18)=h2; break;
                end
            end
            if sl(17)==0
                [sl(17),sl(18)]=sample_pair_sum_stratified(h_lo,h_hi,LB(17),UB(17),TARGET_H_SUM);
            end
            for attempt_h = 1:50
                h3 = h_lo+(h_hi-h_lo)*rand();
                h4 = TARGET_H_SUM-h3;
                if h4>=LB(19)&&h4<=UB(19)&&abs(h3-h4)>=10
                    sl(19)=h3; sl(20)=h4; break;
                end
            end
            if sl(19)==0
                [sl(19),sl(20)]=sample_pair_sum_stratified(h_lo,h_hi,LB(19),UB(19),TARGET_H_SUM);
            end
        else
            [sl(17),sl(18)]=sample_pair_sum_stratified(h_lo,h_hi,LB(17),UB(17),TARGET_H_SUM);
            [sl(19),sl(20)]=sample_pair_sum_stratified(h_lo,h_hi,LB(19),UB(19),TARGET_H_SUM);
        end
        [sl(26),sl(27)]=sample_pair_sum_stratified(w_lo,w_hi,LB(26),UB(26),TARGET_W_SUM);
        % td: 在 h 约束下采样
        h_for_td = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections=rejections+1; continue; end
        [tu_vec,tu_ok]=generate_safe_tu(sl(21:24),sl(25),LB(28:31),UB(28:31),Z_MARGIN);
        if ~tu_ok, rejections=rejections+1; continue; end
        sl(28:31)=tu_vec;
        [d_vec,d_ok]=generate_safe_d(sl(26:27),sl(17:20),LB(9:16),UB(9:16),FOUR_DELTA);
        if ~d_ok, rejections=rejections+1; continue; end
        sl(9:16)=d_vec;
        if check_all_constraints(sl,TARGET_W_SUM,TARGET_H_SUM,FOUR_DELTA,SAFE_MARGIN_TD,Z_MARGIN,delta)
            return;
        end
        rejections=rejections+1;
    end
    error('generate_one_stratified: 无法生成合法样本。');
end

function [sl, rejections] = generate_one_boundary(strategy, ...
    LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
    Z_MARGIN, delta)
    rejections = 0;
    EDGE_LO = 0.05;  EDGE_HI = 0.95;
    for attempt = 1:500
        sl = zeros(1,31);
        sl(21:24) = LB(21:24)+(UB(21:24)-LB(21:24)).*rand(1,4);  % Lu
        sl(25) = 57.0;                                              % Ltot
        [sl(17),sl(18)]=sample_pair_sum(LB(17),UB(17),TARGET_H_SUM);
        [sl(19),sl(20)]=sample_pair_sum(LB(19),UB(19),TARGET_H_SUM);
        [sl(26),sl(27)]=sample_pair_sum(LB(26),UB(26),TARGET_W_SUM);
        % td: 在 h 约束下采样 (strategy 4 之后再推向边界)
        h_for_td = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections=rejections+1; continue; end
        % 策略切换
        switch strategy
            case 1
                for k=[26,27]
                    if rand()<0.5, sl(k)=LB(k)+(UB(k)-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(UB(k)-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
                sl(27)=TARGET_W_SUM-sl(26);
                if sl(27)<LB(27)||sl(27)>UB(27), continue; end
            case 2
                for k=[17,19]
                    if rand()<0.5, sl(k)=LB(k)+(UB(k)-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(UB(k)-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
                sl(18)=TARGET_H_SUM-sl(17); sl(20)=TARGET_H_SUM-sl(19);
                if any(sl(17:20)<LB(17:20))||any(sl(17:20)>UB(17:20)), continue; end
                % h changed, re-constrain td
                h_for_td2 = [sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
                for i = 1:8
                    sl(i) = min(sl(i), h_for_td2(i) - SAFE_MARGIN_TD);
                    if sl(i) < LB(i), td_ok = false; break; end
                end
                if ~td_ok, rejections=rejections+1; continue; end
            case 3
                for k=21:24
                    if rand()<0.5, sl(k)=LB(k)+(UB(k)-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(UB(k)-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
            case 4
                % td → 边界, 但范围受 h 约束
                for k = 1:8
                    actual_ub = min(UB(k), h_for_td(k) - SAFE_MARGIN_TD);
                    if rand()<0.5, sl(k)=LB(k)+(actual_ub-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(actual_ub-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
            case 5
                for k=1:4
                    max_tu=sl(25)-sl(20+k)-Z_MARGIN;
                    if max_tu<LB(27+k), continue; end
                    actual_ub=min(UB(27+k),max_tu);
                    if rand()<0.5, sl(27+k)=LB(27+k)+(actual_ub-LB(27+k))*EDGE_LO*rand();
                    else, sl(27+k)=LB(27+k)+(actual_ub-LB(27+k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
            case 6
                w_for_d=[sl(26);sl(26);sl(27);sl(27);sl(26);sl(26);sl(27);sl(27)];
                h_for_d=[sl(17);sl(18);sl(19);sl(20);sl(17);sl(18);sl(19);sl(20)];
                d_limit=min(min(w_for_d,h_for_d)-FOUR_DELTA, h_for_d*0.65);
                if any(d_limit<LB(9:16)), continue; end
                for k=1:8
                    actual_ub=min(UB(8+k),d_limit(k));
                    if rand()<0.5, sl(8+k)=LB(8+k)+(actual_ub-LB(8+k))*EDGE_LO*rand();
                    else, sl(8+k)=LB(8+k)+(actual_ub-LB(8+k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
            case 7
                for k=21:24
                    if rand()<0.5, sl(k)=LB(k)+(UB(k)-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(UB(k)-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
                for k=1:8
                    actual_ub = min(UB(k), h_for_td(k) - SAFE_MARGIN_TD);
                    if rand()<0.5, sl(k)=LB(k)+(actual_ub-LB(k))*EDGE_LO*rand();
                    else, sl(k)=LB(k)+(actual_ub-LB(k))*(EDGE_HI+(1-EDGE_HI)*rand()); end
                end
        end
        if strategy~=5
            [tu_vec,tu_ok]=generate_safe_tu(sl(21:24),sl(25),LB(28:31),UB(28:31),Z_MARGIN);
            if ~tu_ok, rejections=rejections+1; continue; end
            sl(28:31)=tu_vec;
        end
        if strategy~=6
            [d_vec,d_ok]=generate_safe_d(sl(26:27),sl(17:20),LB(9:16),UB(9:16),FOUR_DELTA);
            if ~d_ok, rejections=rejections+1; continue; end
            sl(9:16)=d_vec;
        end
        if check_all_constraints(sl,TARGET_W_SUM,TARGET_H_SUM,FOUR_DELTA,SAFE_MARGIN_TD,Z_MARGIN,delta)
            return;
        end
        rejections=rejections+1;
    end
    error('generate_one_boundary: 无法生成合法样本。');
end
