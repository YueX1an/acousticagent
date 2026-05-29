% =========================================================================
% generate_strategic_dataset.m
% 战略性数据集生成脚本 — 面向双任务逆向设计
% =========================================================================
%
% 目标任务:
%   任务1: 50-2000Hz 宽带完美吸声结构逆向设计
%   任务2: 400/900/1400/1900Hz 四频段独立调控（全息图案复用）
%
% 数据集组成 (总计约 200,000 样本, 可调整):
%   Part A (40%, 80,000): 全空间随机基准 — 保证参数空间广泛覆盖
%   Part B (30%, 60,000): GA 最优参数邻域微扰 — 强化宽带完美吸声区域
%   Part C (20%, 40,000): 分层多样化 — 针对四频段独立调控能力
%   Part D (10%, 20,000): 边界探索 — 覆盖参数空间极端区域
%
% 31维参数布局:
%   td(1:8)   — 内插管长度 (mm),       每腔每层各1个, 共8个
%   d(9:16)   — 内插管直径/孔径 (mm),  每腔每层各1个, 共8个
%   h(17:20)  — 单腔高度 (mm),         每个腔各1个, 共4个
%   Lu(21:24) — 上层腔体长度 (mm),     每个腔各1个, 共4个
%   Ltot(25)  — 结构总高度 (mm),       固定为57
%   w(26:27)  — 主腔宽度 (mm),         每列各1个, 共2个
%   tu(28:31) — 上层连接管长度 (mm),   每个腔各1个, 共4个
%
% 关键物理约束:
%   1. w1 + w2 = W - (nw+1)*delta = 46.7
%   2. h1 + h2 = h3 + h4 = H - (nh+1)*delta = 46.7
%   3. d(i) <= min(w_col, h_row) - 4*delta  (孔径不能穿出腔体)
%   4. Lu(k) + tu(k) + 3.0 <= Ltot           (Z轴方向管不能穿出)
%   5. td(i) <= h_row - 1.5*delta            (管长不能超过腔高)
%   6. d(i) / h_row <= 0.65                  (管径/腔高比例, 来自外部GA约束)
%   7. h(i) / Ltot <= 0.95                   (腔高/总高比例, 防御性约束)
%
% 依赖:
%   fun_structure_broadband_nlayer_HB_func.m  (声学 TMM 仿真核心)
%   zs_square_grating_multilayer_mnum_amend.m (被仿真函数调用的阻抗计算)
%   Parallel Computing Toolbox (parfor 加速)
%
% 输出文件 (保存在 data/ 目录下):
%   new_dataset_slN.txt    — N×31 结构参数矩阵
%   new_dataset_alphaM.txt — 196×N 吸声系数矩阵 (频率 50:10:2000 Hz)
%
% 使用方法:
%   1. 确保上述 .m 依赖文件在 MATLAB 路径中
%   2. 在 MATLAB 中直接运行本脚本
%   3. 可修改 PART_SIZES 向量调整各部分组成比例
%
% 作者备注:
%   - 基于 best_structure_params_31D.txt 中的 GA 最优解进行微扰
%   - 遵循 generate_31param_data_norma_20260502.m 中最新的参数边界
%   - 采用与 Random_sl_mutilcavs_nlayer_func_31params_fixed_20260502.m
%     一致的物理防穿模约束逻辑
% =========================================================================

clc; clear; close all;

%% ==================== 0. 核心配置 ====================

% --- 数据集规模 (可调整) ---
%     格式: [A_随机基准, B_GA微扰, C_分层多样, D_边界探索]
PART_SIZES = [80000, 60000, 40000, 20000];
N_total = sum(PART_SIZES);

% --- 结构常数 (与最新设计界面 20260502 一致) ---
nlayer = 2;        % 层数 (双层结构)
ncell  = 4;        % 每层腔体数
nw     = 2;        % 宽度方向腔体列数
delta  = 1.0;      % 壁厚 (mm)
H      = 49.7;     % 外部总高 (mm)
W      = 49.7;     % 外部总宽 (mm)
nh     = ncell / nw;  % 每列腔体行数 = 2

% --- 导出约束目标值 ---
TARGET_W_SUM = W - (nw + 1) * delta;        % w1 + w2 = 46.7
TARGET_H_SUM = H - (nh + 1) * delta;        % h1+h2 = h3+h4 = 46.7
FOUR_DELTA   = 4 * delta;                   % = 4.0, d 参数的安全边距
SAFE_MARGIN_TD = 1.5 * delta;               % = 1.5, td 的安全边距
Z_MARGIN       = 3.0;                       % Z轴方向最小余量

% --- 31 维参数全局边界 ---
% 顺序: [td(1:8), d(9:16), h(17:20), Lu(21:24), Ltot(25), w(26:27), tu(28:31)]
LB = [...
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, ...   % td  (1-8)
    1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, ...   % d   (9-16)
   10.0, 10.0, 10.0, 10.0, ...                            % h   (17-20)
   15.0, 15.0, 15.0, 15.0, ...                            % Lu  (21-24)
   57.0, ...                                               % Ltot(25) 固定
   15.0, 15.0, ...                                         % w   (26-27)
    0.0,  0.0,  0.0,  0.0];                               % tu  (28-31)

UB = [...
   200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, ... % td
    30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0, ...  % d
    50.0,  50.0,  50.0,  50.0, ...                               % h
    40.0,  40.0,  40.0,  40.0, ...                               % Lu
    57.0, ...                                                     % Ltot 固定
    40.0,  40.0, ...                                              % w
   200.0, 200.0, 200.0, 200.0];                                  % tu

% --- 仿真配置 ---
fmin   = 50;    fstep = 10;   fmax = 2000;
f0     = fmin:fstep:fmax;     % 196 个频点 (50:10:2000)
fL     = length(f0);
mnum   = 500;                 % 傅里叶模态截断数
Norder = 9;                   % 高阶模态展开阶数

% --- GA 最优参考参数 (来自 best_structure_params_31D.txt) ---
% 这是经过遗传算法 + 人工微扰筛选验证的宽带最优解, 500Hz+ 平均吸声 > 0.90
GA_BEST = [...
    7.4617428,  2.4924932,  1.4229646,  2.8019668,  1.0639180, 14.478907,  1.0000000,  9.3083567, ... % td(1-8)
   11.063156,  10.255968,  10.293949,  11.317911,  19.959003,  8.0606058, 12.831908,  4.9389906, ... % d(9-16)
   26.619550,  20.080450,  17.394773,  29.305227, ...  % h(17-20)
   50.000000,  21.568932,  50.000000,  30.359377, ...  % Lu(21-24) [注意: Lu1,Lu3=50 超出新界40, 将裁剪]
   57.000000, ...                                      % Ltot(25)
   26.315897,  20.384103, ...                          % w(26-27)
    0.011222627, 14.510600, 0.13122252, 15.312971];     % tu(28-31)

% 裁剪 GA 参数至当前全局边界
GA_BEST_CLIPPED = max(LB, min(UB, GA_BEST));
% 还原被裁剪的关键参数 (Lu1, Lu3=50 是 GA 验证有效的, 略微超出新界但物理可行)
% 在微扰时允许探索到原始值附近
GA_Lu_original = GA_BEST(21:24);

% --- 输出路径 ---
OUT_DIR = 'data';
if ~exist(OUT_DIR, 'dir')
    mkdir(OUT_DIR);
end

% 固定随机种子 (可复现)
rng(42);

%% ==================== 1. Part A: 全空间随机基准 (40%) ====================
fprintf('=========================================================\n');
fprintf('  Part A: 生成 %d 个全空间随机基准样本\n', PART_SIZES(1));
fprintf('=========================================================\n');

slN_A = zeros(PART_SIZES(1), 31);
rejected_A = 0;

tic;
parfor i = 1:PART_SIZES(1)
    [sl, rej] = generate_one_random(LB, UB, TARGET_W_SUM, TARGET_H_SUM, ...
        FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta);
    slN_A(i, :) = sl;
    rejected_A = rejected_A + rej;
end
elapsed_A = toc;
fprintf('Part A 完成。耗时: %.1f 秒, 平均拒绝次数: %.1f/样本\n', ...
    elapsed_A, rejected_A / PART_SIZES(1));

%% ==================== 2. Part B: GA 最优邻域微扰 (30%) ====================
fprintf('=========================================================\n');
fprintf('  Part B: 生成 %d 个 GA 最优邻域微扰样本\n', PART_SIZES(2));
fprintf('=========================================================\n');

slN_B = zeros(PART_SIZES(2), 31);
anchor = GA_BEST_CLIPPED;

% 三级噪声: 紧邻域(σ=3%), 中间域(σ=8%), 远邻域(σ=15%)
noise_levels  = [0.03, 0.08, 0.15];
noise_weights = [0.50, 0.30, 0.20];
noise_cumsum  = cumsum(noise_weights);

tic;
parfor i = 1:PART_SIZES(2)
    % 按权重随机选择噪声等级
    r = rand();
    if r < noise_cumsum(1)
        sigma_ratio = noise_levels(1);
    elseif r < noise_cumsum(2)
        sigma_ratio = noise_levels(2);
    else
        sigma_ratio = noise_levels(3);
    end

    [sl, ~] = perturb_around_anchor(anchor, GA_Lu_original, sigma_ratio, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
        Z_MARGIN, delta);
    slN_B(i, :) = sl;
end
elapsed_B = toc;
fprintf('Part B 完成。耗时: %.1f 秒 (%.1f 样本/秒)\n', ...
    elapsed_B, PART_SIZES(2) / elapsed_B);

%% ==================== 3. Part C: 分层多样化 (20%) ====================
% 策略: 刻意在参数空间的不同区域采样
%   Profile 1: 高 h + 高 w → 允许大孔径, 偏向宽带吸收
%   Profile 2: 低 h + 高 w → 非对称, 偏向低频调控
%   Profile 3: 高 h + 低 w → 非对称, 偏向高频调控
%   Profile 4: 中等全范围 → 基准覆盖
%   Profile 5: 差异化 h → 增强频率选择性
fprintf('=========================================================\n');
fprintf('  Part C: 生成 %d 个分层多样化样本\n', PART_SIZES(3));
fprintf('=========================================================\n');

slN_C = zeros(PART_SIZES(3), 31);

% 5 种 Profile 的 h 和 w 子范围
prof_h_range = {[30, 50], [10, 25], [30, 50], [10, 50], [10, 50]};
prof_w_range = {[25, 40], [25, 40], [15, 25], [15, 40], [15, 40]};
prof_names   = {'高h+高w','低h+高w','高h+低w','均匀全范围','差异化h'};

samples_per_prof = floor(PART_SIZES(3) / 5);
% 多余样本追加到 Profile 1
prof_counts = samples_per_prof * ones(1, 5);
prof_counts(1) = prof_counts(1) + mod(PART_SIZES(3), 5);

% 预计算每个 Profile 的输出起始索引
prof_start_idx = [1, cumsum(prof_counts(1:end-1)) + 1];

tic;
for prof = 1:5
    n_prof = prof_counts(prof);
    start_i = prof_start_idx(prof);
    fprintf('  Profile %d (%s): %d 样本\n', prof, prof_names{prof}, n_prof);

    h_lo = prof_h_range{prof}(1);
    h_hi = prof_h_range{prof}(2);
    w_lo = prof_w_range{prof}(1);
    w_hi = prof_w_range{prof}(2);
    is_diff_h = (prof == 5);

    parfor j = 1:n_prof
        [sl, ~] = generate_one_stratified(h_lo, h_hi, w_lo, w_hi, is_diff_h, ...
            LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
            Z_MARGIN, delta);
        sl_temp(j, :) = sl;
    end
    slN_C(start_i : start_i + n_prof - 1, :) = sl_temp;
end
elapsed_C = toc;
fprintf('Part C 完成。耗时: %.1f 秒\n', elapsed_C);

%% ==================== 4. Part D: 边界探索 (10%) ====================
% 策略: 将部分参数推向边界极值 (5%边缘区域), 确保模型见过参数空间"墙角"
fprintf('=========================================================\n');
fprintf('  Part D: 生成 %d 个边界探索样本\n', PART_SIZES(4));
fprintf('=========================================================\n');

slN_D = zeros(PART_SIZES(4), 31);
N_STRATEGIES = 7;  % 7 种边界策略循环

tic;
parfor i = 1:PART_SIZES(4)
    strategy = mod(i - 1, N_STRATEGIES) + 1;
    [sl, ~] = generate_one_boundary(strategy, ...
        LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
        Z_MARGIN, delta);
    slN_D(i, :) = sl;
end
elapsed_D = toc;
fprintf('Part D 完成。耗时: %.1f 秒\n', elapsed_D);

%% ==================== 5. 合并并打乱数据集 ====================
fprintf('=========================================================\n');
fprintf('  合并与打乱数据集...\n');
fprintf('=========================================================\n');

slN_all = [slN_A; slN_B; slN_C; slN_D];
shuffle_idx = randperm(size(slN_all, 1));
slN = slN_all(shuffle_idx, :);

clear slN_A slN_B slN_C slN_D slN_all;

fprintf('合并后总样本数: %d\n', size(slN, 1));
fprintf('  - Part A (随机基准): %d\n', PART_SIZES(1));
fprintf('  - Part B (GA 微扰):  %d\n', PART_SIZES(2));
fprintf('  - Part C (分层多样): %d\n', PART_SIZES(3));
fprintf('  - Part D (边界探索): %d\n', PART_SIZES(4));

%% ==================== 6. 大规模并行仿真 ====================
fprintf('=========================================================\n');
fprintf('  启动大规模并行 TMM 仿真...\n');
fprintf('  样本数: %d, 频点数: %d (50:10:2000 Hz)\n', N_total, fL);
fprintf('  预计耗时取决于 CPU 核心数和样本量...\n');
fprintf('=========================================================\n');

alphaM = zeros(fL, N_total);

tic;
parfor n = 1:N_total
    sl = slN(n, :);
    try
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
            f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
        alphaM(:, n) = alpha(:);
    catch ME
        % 仿真异常时填入 NaN, 后续统一剔除
        alphaM(:, n) = NaN(fL, 1);
    end

    if mod(n, 10000) == 0
        fprintf('  仿真进度: %d / %d (%.1f%%)\n', n, N_total, 100 * n / N_total);
    end
end
elapsed_sim = toc;

fprintf('仿真完成。总耗时: %.1f 分钟 (平均 %.2f 秒/样本)\n', ...
    elapsed_sim / 60, elapsed_sim / N_total);

%% ==================== 7. 后处理: 剔除仿真失败样本 ====================
fprintf('=========================================================\n');
fprintf('  后处理: 剔除仿真失败样本...\n');
fprintf('=========================================================\n');

valid_cols = all(isfinite(alphaM), 1);
n_invalid = sum(~valid_cols);

if n_invalid > 0
    fprintf('⚠️  检测到 %d 个仿真失败样本 (NaN/Inf), 正在剔除...\n', n_invalid);
    slN = slN(valid_cols, :);
    alphaM = alphaM(:, valid_cols);
    fprintf('   剔除后有效样本数: %d\n', size(slN, 1));
else
    fprintf('✅ 所有样本仿真成功, 无剔除。\n');
end

%% ==================== 8. 快速质量评估 ====================
fprintf('=========================================================\n');
fprintf('  数据集快速质量评估...\n');
fprintf('=========================================================\n');

% 8a. 宽带吸声统计
mean_abs_all     = mean(alphaM, 1);
mean_abs_500plus = mean(alphaM(f0 >= 500, :), 1);
std_abs_500plus  = std(alphaM(f0 >= 500, :), 0, 1);
[best_broadband_val, best_broadband_idx] = max(mean_abs_500plus);

fprintf('\n--- 宽带吸声统计 (任务1 相关) ---\n');
fprintf('全频段 (50-2000Hz):\n');
fprintf('  平均吸声率:       %.4f ± %.4f\n', mean(mean_abs_all), std(mean_abs_all));
fprintf('  最高单样本均值:   %.4f\n', max(mean_abs_all));
fprintf('500Hz+ 宽带平坦度:\n');
fprintf('  平均吸声率:       %.4f ± %.4f\n', mean(mean_abs_500plus), std(mean_abs_500plus));
fprintf('  最高单样本均值:   %.4f (样本 #%d)\n', best_broadband_val, best_broadband_idx);
fprintf('  最优波动 (std):   %.4f\n', min(std_abs_500plus));

% 8b. 四频段调控能力统计
target_freqs = [400, 900, 1400, 1900];
target_idx = round((target_freqs - fmin) / fstep) + 1;

fprintf('\n--- 四频段吸声率分布 (任务2 相关) ---\n');
fprintf('  频率    最小值   最大值   均值     标准差\n');
for i = 1:4
    abs_at_freq = alphaM(target_idx(i), :);
    fprintf('  %4d Hz  %.3f   %.3f   %.3f   %.3f\n', ...
        target_freqs(i), min(abs_at_freq), max(abs_at_freq), ...
        mean(abs_at_freq), std(abs_at_freq));
end

% 8c. 四频段联合覆盖: 统计有多少样本能在各频段同时达到高低吸收
fprintf('\n--- 四频段二元组合覆盖 (高低阈值=0.6) ---\n');
high_mask = alphaM(target_idx, :) > 0.6;
low_mask  = alphaM(target_idx, :) < 0.4;
% 统计一些有代表性的组合
combos = {...
    '全高 [H H H H]', [1,1,1,1]; ...
    '全低 [L L L L]', [0,0,0,0]; ...
    '[H L L L]',      [1,0,0,0]; ...
    '[L H L L]',      [0,1,0,0]; ...
    '[L L H L]',      [0,0,1,0]; ...
    '[L L L H]',      [0,0,0,1]};
for c = 1:size(combos, 1)
    if all(combos{c, 2} == 1)
        count = sum(all(high_mask, 1));
    elseif all(combos{c, 2} == 0)
        count = sum(all(low_mask, 1));
    else
        % 混合模式: 高吸收位用 high_mask, 低吸收位用 low_mask
        m = ones(1, size(high_mask, 2));
        for fi = 1:4
            if combos{c, 2}(fi) == 1
                m = m & high_mask(fi, :);
            else
                m = m & low_mask(fi, :);
            end
        end
        count = sum(m);
    end
    fprintf('  %s: %d 样本\n', combos{c, 1}, count);
end

% 8d. 参数空间覆盖检查
fprintf('\n--- 参数空间实际覆盖 vs 允许范围 ---\n');
for i = 1:31
    fprintf('  X%02d: 实际 [%.2f, %.2f]  |  允许 [%.2f, %.2f]\n', ...
        i, min(slN(:, i)), max(slN(:, i)), LB(i), UB(i));
end

%% ==================== 9. 保存 ====================
fprintf('=========================================================\n');
fprintf('  保存数据集至 data/ 目录...\n');
fprintf('=========================================================\n');

save(fullfile(OUT_DIR, 'new_dataset_slN.txt'), 'slN', '-ascii', '-double');
save(fullfile(OUT_DIR, 'new_dataset_alphaM.txt'), 'alphaM', '-ascii', '-double');

fprintf('✅ 数据集保存完成!\n');
fprintf('   结构参数:  data/new_dataset_slN.txt    (%d x 31)\n', size(slN, 1));
fprintf('   吸声系数:  data/new_dataset_alphaM.txt (%d x %d)\n', ...
    size(alphaM, 1), size(alphaM, 2));
fprintf('\n');
fprintf('   最终有效样本数: %d / %d (%.1f%% 有效率)\n', ...
    size(slN, 1), N_total, 100 * size(slN, 1) / N_total);
fprintf('   频率范围:       %d - %d Hz (步长 %d Hz, 共 %d 频点)\n', ...
    fmin, fmax, fstep, fL);
fprintf('   参数维度:       31\n');
fprintf('\n');
fprintf('=========================================================\n');
fprintf('  脚本执行完毕。\n');
fprintf('  可使用 Python create_new_dataset.py 将数据划分为\n');
fprintf('  train/val/test 三个 .npz 子集。\n');
fprintf('=========================================================\n');

%% ==================== 10. 局部函数定义 ====================
% MATLAB 脚本中, 所有局部函数必须放在脚本末尾

% =========================================================================
% 函数: sample_pair_sum
% 生成一对参数 (v1, v2), 满足 v1+v2 = target_sum 且均在 [lb, ub] 范围内
% =========================================================================
function [v1, v2] = sample_pair_sum(lb, ub, target_sum)
    for attempt = 1:200
        v1 = lb + (ub - lb) * rand();
        v2 = target_sum - v1;
        if v2 >= lb && v2 <= ub
            return;
        end
    end
    % 回退方案: 取中值
    v1 = target_sum / 2;
    v2 = target_sum / 2;
    v1 = max(lb, min(ub, v1));
    v2 = max(lb, min(ub, v2));
end

% =========================================================================
% 函数: sample_pair_sum_stratified
% 分层采样: v1 在 [lb_primary, ub_primary] 内, v2 只需在 [lb_other, ub_other] 内
% =========================================================================
function [v1, v2] = sample_pair_sum_stratified(lb_primary, ub_primary, lb_other, ub_other, target_sum)
    for attempt = 1:200
        v1 = lb_primary + (ub_primary - lb_primary) * rand();
        v2 = target_sum - v1;
        if v2 >= lb_other && v2 <= ub_other
            return;
        end
    end
    v1 = (lb_primary + ub_primary) / 2;
    v2 = target_sum - v1;
    v1 = max(lb_primary, min(ub_primary, v1));
    v2 = max(lb_other, min(ub_other, v2));
end

% =========================================================================
% 函数: check_all_constraints
% 全面检查一个 31 维参数向量是否满足所有物理约束
% 返回值: ok=true 表示合法, ok=false 表示非法
% =========================================================================
function ok = check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
    FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    h  = sl(17:20);
    w  = sl(26:27);
    d  = sl(9:16);
    Lu = sl(21:24);
    tu = sl(28:31);
    Ltot = sl(25);
    td = sl(1:8);

    % 约束1: w1 + w2 = TARGET_W_SUM
    if abs(sum(w) - TARGET_W_SUM) > 0.02
        ok = false; return;
    end

    % 约束2: h1+h2 = h3+h4 = TARGET_H_SUM
    if abs(h(1) + h(2) - TARGET_H_SUM) > 0.02 || ...
       abs(h(3) + h(4) - TARGET_H_SUM) > 0.02
        ok = false; return;
    end

    % 约束3: d(i) <= min(w_col_i, h_row_i) - 4*delta
    % 腔体→列映射: 腔1,2→w1, 腔3,4→w2 (每层相同)
    w_for_d = [w(1); w(1); w(2); w(2); w(1); w(1); w(2); w(2)];
    h_for_d = [h(1); h(2); h(3); h(4); h(1); h(2); h(3); h(4)];
    d_max_allowed = min(w_for_d, h_for_d) - FOUR_DELTA;
    if any(d(:) > d_max_allowed)
        ok = false; return;
    end

    % 约束4: Z轴防穿模 Lu(k) + tu(k) + Z_MARGIN <= Ltot
    for k = 1:4
        if Lu(k) + tu(k) + Z_MARGIN > Ltot
            ok = false; return;
        end
    end

    % 约束5: td(i) <= h_row_i - SAFE_MARGIN_TD
    td_h = [h(1); h(2); h(3); h(4); h(1); h(2); h(3); h(4)];
    if any(td(:) > td_h - SAFE_MARGIN_TD)
        ok = false; return;
    end

    % 约束6: d(i) / h_row_i <= 0.65 (管径/腔高比例, 来自外部GA约束)
    if any(d(:) ./ h_for_d > 0.65)
        ok = false; return;
    end

    % 约束7: h(i) / Ltot <= 0.95 (腔高/总高比例, 防御性约束)
    if any(h(:) / Ltot > 0.95)
        ok = false; return;
    end

    ok = true;
end

% =========================================================================
% 函数: generate_safe_d
% 根据当前 w 和 h 在物理允许范围内安全生成 d 参数向量(8×1)
% 返回合法的 d 向量, 以及是否成功 (success_flag)
% =========================================================================
function [d_vec, success] = generate_safe_d(w, h, LB_d, UB_d, FOUR_DELTA)
    w_for_d = [w(1); w(1); w(2); w(2); w(1); w(1); w(2); w(2)];
    h_for_d = [h(1); h(2); h(3); h(4); h(1); h(2); h(3); h(4)];
    d_limit = min(min(w_for_d, h_for_d) - FOUR_DELTA, h_for_d * 0.65);

    if any(d_limit < LB_d)
        d_vec = zeros(8, 1);
        success = false;
        return;
    end

    d_vec = zeros(8, 1);
    for i = 1:8
        actual_ub = min(UB_d(i), d_limit(i));
        d_vec(i) = LB_d(i) + (actual_ub - LB_d(i)) * rand();
    end
    success = true;
end

% =========================================================================
% 函数: generate_safe_tu
% 根据当前 Lu 和 Ltot 安全生成 tu 参数向量(4×1)
% =========================================================================
function [tu_vec, success] = generate_safe_tu(Lu, Ltot, LB_tu, UB_tu, Z_MARGIN)
    tu_vec = zeros(4, 1);
    for k = 1:4
        max_tu = Ltot - Lu(k) - Z_MARGIN;
        if max_tu < LB_tu(k)
            success = false; return;
        end
        actual_ub = min(UB_tu(k), max_tu);
        tu_vec(k) = LB_tu(k) + (actual_ub - LB_tu(k)) * rand();
    end
    success = true;
end

% =========================================================================
% 函数: generate_one_random
% 生成一个完全随机的、100% 物理合法的 31 维参数向量
% 返回: [sl(1×31), 内部拒绝次数]
% =========================================================================
function [sl, rejections] = generate_one_random(LB, UB, TARGET_W_SUM, ...
    TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
    rejections = 0;
    for attempt = 1:500
        sl = zeros(1, 31);

        % 1. 独立参数: Lu (全范围随机)
        sl(21:24) = LB(21:24) + (UB(21:24) - LB(21:24)) .* rand(1, 4);

        % 2. Ltot 固定在 57
        sl(25) = 57.0;

        % 3. 耦合参数: h (满足对偶总和约束)
        [sl(17), sl(18)] = sample_pair_sum(LB(17), UB(17), TARGET_H_SUM);
        [sl(19), sl(20)] = sample_pair_sum(LB(19), UB(19), TARGET_H_SUM);

        % 4. 耦合参数: w (满足总和约束)
        [sl(26), sl(27)] = sample_pair_sum(LB(26), UB(26), TARGET_W_SUM);

        % 5. td: 在 h 约束下采样 (td <= h - 1.5*delta)
        h_for_td = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections = rejections + 1; continue; end

        % 6. Z轴耦合: tu
        [tu_vec, tu_ok] = generate_safe_tu(sl(21:24), sl(25), ...
            LB(28:31), UB(28:31), Z_MARGIN);
        if ~tu_ok
            rejections = rejections + 1; continue;
        end
        sl(28:31) = tu_vec;

        % 7. 依赖参数: d
        [d_vec, d_ok] = generate_safe_d(sl(26:27), sl(17:20), ...
            LB(9:16), UB(9:16), FOUR_DELTA);
        if ~d_ok
            rejections = rejections + 1; continue;
        end
        sl(9:16) = d_vec;

        % 8. 最终全量检查
        if check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
                FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
            return;
        end
        rejections = rejections + 1;
    end
    error('generate_one_random: 无法生成合法样本, 请检查参数边界。');
end

% =========================================================================
% 函数: perturb_around_anchor
% 在锚点参数周围加上高斯噪声微扰, 生成一个 100% 物理合法的变体
% 采用零和微扰保持总和约束, 动态计算 d 和 tu 的安全范围
% =========================================================================
function [sl, success] = perturb_around_anchor(anchor, Lu_original, ...
    sigma_ratio, LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta)

    success = true;
    for attempt = 1:500
        sl = anchor;

        % --- 独立参数: 高斯噪声微扰 (td, Lu) ---
        for idx = [1:8, 21:24]
            sigma = max(abs(anchor(idx)) * sigma_ratio, 0.3);
            sl(idx) = anchor(idx) + sigma * randn();
            sl(idx) = max(LB(idx), min(UB(idx), sl(idx)));
        end

        % --- 耦合参数 h: 零和微扰 ---
        % h1,h2
        dh12 = anchor(17) * sigma_ratio * (2 * rand() - 1);
        h1_new = anchor(17) + dh12;
        h2_new = anchor(18) - dh12;
        if h1_new >= LB(17) && h1_new <= UB(17) && h2_new >= LB(18) && h2_new <= UB(18)
            sl(17) = h1_new; sl(18) = h2_new;
        end
        % h3,h4
        dh34 = anchor(19) * sigma_ratio * (2 * rand() - 1);
        h3_new = anchor(19) + dh34;
        h4_new = anchor(20) - dh34;
        if h3_new >= LB(19) && h3_new <= UB(19) && h4_new >= LB(20) && h4_new <= UB(20)
            sl(19) = h3_new; sl(20) = h4_new;
        end

        % --- 耦合参数 w: 零和微扰 ---
        dw = anchor(26) * sigma_ratio * (2 * rand() - 1);
        w1_new = anchor(26) + dw;
        w2_new = anchor(27) - dw;
        if w1_new >= LB(26) && w1_new <= UB(26) && w2_new >= LB(27) && w2_new <= UB(27)
            sl(26) = w1_new; sl(27) = w2_new;
        end

        % --- Z轴耦合: tu (动态上限) ---
        tu_ok = true;
        for k = 1:4
            sigma = max(abs(anchor(27 + k)) * sigma_ratio, 0.3);
            tu_candidate = anchor(27 + k) + sigma * randn();
            tu_candidate = max(LB(27 + k), tu_candidate);
            max_tu = sl(25) - sl(20 + k) - Z_MARGIN;
            sl(27 + k) = min(tu_candidate, max_tu);
            if sl(27 + k) < LB(27 + k)
                tu_ok = false; break;
            end
        end
        if ~tu_ok, continue; end

        % --- 依赖参数 d: 动态安全上限 ---
        [d_vec, d_ok] = generate_safe_d(sl(26:27), sl(17:20), ...
            LB(9:16), UB(9:16), FOUR_DELTA);
        if ~d_ok, continue; end
        % 倾向于锚点附近的 d 值
        for j = 1:8
            sigma = max(abs(anchor(8 + j)) * sigma_ratio, 0.3);
            d_candidate = anchor(8 + j) + sigma * randn();
            d_candidate = max(LB(8 + j), min(d_vec(j), d_candidate));
            if d_candidate <= d_vec(j)
                sl(8 + j) = d_candidate;
            else
                sl(8 + j) = d_vec(j);
            end
        end

        % --- 管长 td 几何约束: td(i) <= h_row - 1.5*delta ---
        h_rep = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
        for j = 1:8
            max_td = h_rep(j) - SAFE_MARGIN_TD;
            if sl(j) > max_td
                sl(j) = max(LB(j), max_td);
            end
        end

        % 最终全量检查
        if check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
                FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
            return;
        end
    end
    % 极端情况下回退到锚点本身
    sl = anchor;
    success = false;
end

% =========================================================================
% 函数: generate_one_stratified
% 在指定的 h/w 子范围内生成一个分层多样化样本
% is_diff_h=true 时刻意让各腔高度差异化
% =========================================================================
function [sl, rejections] = generate_one_stratified(h_lo, h_hi, w_lo, w_hi, ...
    is_diff_h, LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, ...
    SAFE_MARGIN_TD, Z_MARGIN, delta)
    rejections = 0;
    for attempt = 1:500
        sl = zeros(1, 31);

        % 独立参数 (全范围)
        sl(21:24) = LB(21:24) + (UB(21:24) - LB(21:24)) .* rand(1, 4);
        sl(25)    = 57.0;

        % h: 在指定子范围 [h_lo, h_hi] 内采样
        if is_diff_h
            % 刻意差异化: 确保 |h1 - h2| >= 10 且 |h3 - h4| >= 10
            for attempt_h = 1:50
                h1 = h_lo + (h_hi - h_lo) * rand();
                h2 = TARGET_H_SUM - h1;
                if h2 >= LB(17) && h2 <= UB(17) && abs(h1 - h2) >= 10
                    sl(17) = h1; sl(18) = h2; break;
                end
            end
            if sl(17) == 0  % h1==0 表示未初始化
                [sl(17), sl(18)] = sample_pair_sum_stratified(h_lo, h_hi, LB(17), UB(17), TARGET_H_SUM);
            end
            for attempt_h = 1:50
                h3 = h_lo + (h_hi - h_lo) * rand();
                h4 = TARGET_H_SUM - h3;
                if h4 >= LB(19) && h4 <= UB(19) && abs(h3 - h4) >= 10
                    sl(19) = h3; sl(20) = h4; break;
                end
            end
            if sl(19) == 0
                [sl(19), sl(20)] = sample_pair_sum_stratified(h_lo, h_hi, LB(19), UB(19), TARGET_H_SUM);
            end
        else
            [sl(17), sl(18)] = sample_pair_sum_stratified(h_lo, h_hi, LB(17), UB(17), TARGET_H_SUM);
            [sl(19), sl(20)] = sample_pair_sum_stratified(h_lo, h_hi, LB(19), UB(19), TARGET_H_SUM);
        end

        % w: 在指定子范围 [w_lo, w_hi] 内采样 (补足值使用全局范围)
        [sl(26), sl(27)] = sample_pair_sum_stratified(w_lo, w_hi, LB(26), UB(26), TARGET_W_SUM);

        % td: 在 h 约束下采样
        h_for_td = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections = rejections + 1; continue; end

        % tu: Z轴安全
        [tu_vec, tu_ok] = generate_safe_tu(sl(21:24), sl(25), ...
            LB(28:31), UB(28:31), Z_MARGIN);
        if ~tu_ok
            rejections = rejections + 1; continue;
        end
        sl(28:31) = tu_vec;

        % d: 物理安全
        [d_vec, d_ok] = generate_safe_d(sl(26:27), sl(17:20), ...
            LB(9:16), UB(9:16), FOUR_DELTA);
        if ~d_ok
            rejections = rejections + 1; continue;
        end
        sl(9:16) = d_vec;

        if check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
                FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
            return;
        end
        rejections = rejections + 1;
    end
    error('generate_one_stratified: 无法生成合法样本。');
end

% =========================================================================
% 函数: generate_one_boundary
% 将指定参数组推向边界极值区域 (下边界 5% 或上边界 5%)
% strategy 1-7 对应不同的参数组
% =========================================================================
function [sl, rejections] = generate_one_boundary(strategy, ...
    LB, UB, TARGET_W_SUM, TARGET_H_SUM, FOUR_DELTA, SAFE_MARGIN_TD, ...
    Z_MARGIN, delta)
    rejections = 0;
    EDGE_LO = 0.05;   % 下边缘区域比例
    EDGE_HI = 0.95;   % 上边缘起始比例

    for attempt = 1:500
        sl = zeros(1, 31);

        % 基础填充: 先生成 Lu, Ltot, h, w, 再生成依赖参数
        sl(21:24) = LB(21:24) + (UB(21:24) - LB(21:24)) .* rand(1, 4);
        sl(25)    = 57.0;
        [sl(17), sl(18)] = sample_pair_sum(LB(17), UB(17), TARGET_H_SUM);
        [sl(19), sl(20)] = sample_pair_sum(LB(19), UB(19), TARGET_H_SUM);
        [sl(26), sl(27)] = sample_pair_sum(LB(26), UB(26), TARGET_W_SUM);

        % 根据策略将指定参数推向边界
        switch strategy
            case 1  % w → 边界
                for k = [26, 27]
                    if rand() < 0.5
                        sl(k) = LB(k) + (UB(k) - LB(k)) * EDGE_LO * rand();
                    else
                        sl(k) = LB(k) + (UB(k) - LB(k)) * (EDGE_HI + (1 - EDGE_HI) * rand());
                    end
                end
                sl(27) = TARGET_W_SUM - sl(26);
                if sl(27) < LB(27) || sl(27) > UB(27), continue; end

            case 2  % h → 边界
                for k = [17, 19]
                    if rand() < 0.5
                        sl(k) = LB(k) + (UB(k) - LB(k)) * EDGE_LO * rand();
                    else
                        sl(k) = LB(k) + (UB(k) - LB(k)) * (EDGE_HI + (1 - EDGE_HI) * rand());
                    end
                end
                sl(18) = TARGET_H_SUM - sl(17);
                sl(20) = TARGET_H_SUM - sl(19);
                if any(sl(17:20) < LB(17:20)) || any(sl(17:20) > UB(17:20)), continue; end

            case 3  % Lu → 边界
                for k = 21:24
                    if rand() < 0.5
                        sl(k) = LB(k) + (UB(k) - LB(k)) * EDGE_LO * rand();
                    else
                        sl(k) = LB(k) + (UB(k) - LB(k)) * (EDGE_HI + (1 - EDGE_HI) * rand());
                    end
                end

            case 5  % tu → 边界
                for k = 1:4
                    max_tu = sl(25) - sl(20 + k) - Z_MARGIN;
                    if max_tu < LB(27 + k), continue; end
                    actual_ub = min(UB(27 + k), max_tu);
                    if rand() < 0.5
                        sl(27 + k) = LB(27 + k) + (actual_ub - LB(27 + k)) * EDGE_LO * rand();
                    else
                        sl(27 + k) = LB(27 + k) + (actual_ub - LB(27 + k)) * ...
                            (EDGE_HI + (1 - EDGE_HI) * rand());
                    end
                end

            case 6  % d → 边界
                w_for_d = [sl(26); sl(26); sl(27); sl(27); sl(26); sl(26); sl(27); sl(27)];
                h_for_d = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
                d_limit = min(min(w_for_d, h_for_d) - FOUR_DELTA, h_for_d * 0.65);
                if any(d_limit < LB(9:16)), continue; end
                for k = 1:8
                    actual_ub = min(UB(8 + k), d_limit(k));
                    if rand() < 0.5
                        sl(8 + k) = LB(8 + k) + (actual_ub - LB(8 + k)) * EDGE_LO * rand();
                    else
                        sl(8 + k) = LB(8 + k) + (actual_ub - LB(8 + k)) * ...
                            (EDGE_HI + (1 - EDGE_HI) * rand());
                    end
                end
        end

        % td: 在 h 约束下采样 (strategies 4,7 在此之后推向边界)
        h_for_td = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
        td_ok = true;
        for i = 1:8
            actual_ub = min(UB(i), h_for_td(i) - SAFE_MARGIN_TD);
            if actual_ub < LB(i), td_ok = false; break; end
            sl(i) = LB(i) + (actual_ub - LB(i)) * rand();
        end
        if ~td_ok, rejections = rejections + 1; continue; end

        % strategy 2 改变了 h → 重新约束 td
        if strategy == 2
            h_for_td2 = [sl(17); sl(18); sl(19); sl(20); sl(17); sl(18); sl(19); sl(20)];
            for i = 1:8
                sl(i) = min(sl(i), h_for_td2(i) - SAFE_MARGIN_TD);
                if sl(i) < LB(i), td_ok = false; break; end
            end
            if ~td_ok, rejections = rejections + 1; continue; end
        end

        % strategies 4 和 7: td → 边界 (范围受 h 约束)
        if strategy == 4 || strategy == 7
            for k = 1:8
                actual_ub = min(UB(k), h_for_td(k) - SAFE_MARGIN_TD);
                if rand() < 0.5
                    sl(k) = LB(k) + (actual_ub - LB(k)) * EDGE_LO * rand();
                else
                    sl(k) = LB(k) + (actual_ub - LB(k)) * (EDGE_HI + (1 - EDGE_HI) * rand());
                end
            end
        end

        % strategy 7: Lu → 边界
        if strategy == 7
            for k = 21:24
                if rand() < 0.5
                    sl(k) = LB(k) + (UB(k) - LB(k)) * EDGE_LO * rand();
                else
                    sl(k) = LB(k) + (UB(k) - LB(k)) * (EDGE_HI + (1 - EDGE_HI) * rand());
                end
            end
        end

        % 补充未被策略处理的依赖参数
        % tu (如果策略不是 5)
        if strategy ~= 5
            [tu_vec, tu_ok] = generate_safe_tu(sl(21:24), sl(25), ...
                LB(28:31), UB(28:31), Z_MARGIN);
            if ~tu_ok
                rejections = rejections + 1; continue;
            end
            sl(28:31) = tu_vec;
        end

        % d (如果策略不是 6)
        if strategy ~= 6
            [d_vec, d_ok] = generate_safe_d(sl(26:27), sl(17:20), ...
                LB(9:16), UB(9:16), FOUR_DELTA);
            if ~d_ok
                rejections = rejections + 1; continue;
            end
            sl(9:16) = d_vec;
        end

        if check_all_constraints(sl, TARGET_W_SUM, TARGET_H_SUM, ...
                FOUR_DELTA, SAFE_MARGIN_TD, Z_MARGIN, delta)
            return;
        end
        rejections = rejections + 1;
    end
    error('generate_one_boundary: 无法生成合法样本。');
end
