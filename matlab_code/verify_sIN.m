% verify_slN.m
% 验证 V2 数据集的数学总和、物理下限以及空间干涉（穿模）情况

clc; clear;
disp('======================================================');
disp('开始对数据集进行全方位物理与数学规范体检...');
disp('======================================================');

% --- 1. 读取数据 ---
filename = '2layer_2_2_31params_10Hz_slN.txt';
if ~isfile(filename)
    error('未找到文件 %s，请检查文件名或路径！', filename);
end
slN = load(filename);
[N_samples, N_params] = size(slN);
fprintf('检测到样本总数: %d， 参数维度: %d\n\n', N_samples, N_params);

if N_params ~= 31
    error('参数维度错误！应为 31，实际为 %d', N_params);
end

% --- 2. 核心物理参数配置 ---
nw = 2; ncell = 4; nlayer = 2; delta = 0.7;
W_total = 49.4; H_total = 49.4;
safe_margin = 1.5 * delta; % 1.05
nh = ncell / nw; % 每列 2 个腔体

% --- 3. 提取各个参数矩阵 ---
% 根据拼接顺序: [td(8), d(8), h(4), Lu(4), Ltot(1), w(2), tu(4)]
td_mat   = slN(:, 1:8);
d_mat    = slN(:, 9:16);
h_mat    = slN(:, 17:20);
Lu_mat   = slN(:, 21:24);
Ltot_mat = slN(:, 25);
w_mat    = slN(:, 26:27);
tu_mat   = slN(:, 28:31);

tol = 1e-4; % 浮点数误差容忍度

% ==========================================
% 检查项 1：W 的数学总和约束
% ==========================================
target_sum_w = W_total - (nw + 1) * delta; % 47.3
actual_sum_w = sum(w_mat, 2);
error_w = max(abs(actual_sum_w - target_sum_w));

if error_w < tol
    fprintf('[通过] W 横向总宽度约束: 最大误差 %.2e\n', error_w);
else
    fprintf('[失败] W 总和约束未通过！最大误差: %f\n', error_w);
end

% ==========================================
% 检查项 2：H 的按列数学总和约束 (修复后的 2x2 拓扑)
% ==========================================
target_sum_h = H_total - (nh + 1) * delta; % 49.4 - 3*0.7 = 47.3
% 左列高度和: h1 + h2
actual_sum_h_col1 = sum(h_mat(:, 1:2), 2);
% 右列高度和: h3 + h4
actual_sum_h_col2 = sum(h_mat(:, 3:4), 2);

error_h1 = max(abs(actual_sum_h_col1 - target_sum_h));
error_h2 = max(abs(actual_sum_h_col2 - target_sum_h));

if max(error_h1, error_h2) < tol
    fprintf('[通过] H 纵向双列独立高度约束: 最大误差 %.2e\n', max(error_h1, error_h2));
else
    fprintf('[失败] H 按列总和约束未通过！Col1误差: %f, Col2误差: %f\n', error_h1, error_h2);
end

% ==========================================
% 检查项 3：基础下限检查 (不能有负数或非法微小值)
% ==========================================
chk_w  = min(w_mat(:)) >= 4.0 - tol;
chk_h  = min(h_mat(:)) >= 4.0 - tol;
chk_d  = min(d_mat(:)) >= 1.0 - tol;
chk_td = min(td_mat(:)) >= 1.0 - tol;
chk_tu = min(tu_mat(:)) >= 0.0 - tol; % tu 的下限是 0
chk_Lu = min(Lu_mat(:)) >= 1.0 - tol;

if chk_w && chk_h && chk_d && chk_td && chk_tu && chk_Lu
    fprintf('[通过] 基础参数下限检查 (无非法微小值或负数)\n');
else
    fprintf('[失败] 基础参数下限检查未通过，请检查是否有参数越界！\n');
end

% ==========================================
% 检查项 4：空间干涉与穿模检测 (极度严格)
% ==========================================
interfere_d = 0; interfere_td = 0; interfere_tu = 0; interfere_Lu = 0;

for i = 1:N_samples
    w_curr = w_mat(i, :);   % 1x2
    h_curr = h_mat(i, :);   % 1x4
    d_curr = d_mat(i, :);   % 1x8
    td_curr = td_mat(i, :); % 1x8
    tu_curr = tu_mat(i, :); % 1x4
    Lu_curr = Lu_mat(i, :); % 1x4
    
    % 重构腔体映射关系
    % w 映射到 4 个腔体: [w1, w1, w2, w2]
    w_rep_4 = [w_curr(1), w_curr(1), w_curr(2), w_curr(2)];
    % 针对两层的 8 个参数映射
    w_rep_8 = [w_rep_4, w_rep_4];
    h_rep_8 = [h_curr, h_curr];
    
    % --- 穿模判定逻辑 ---
    % 1. d 不能超过 min(w,h) - 4*delta
    limit_d = min(w_rep_8, h_rep_8) - 4 * delta;
    if any(d_curr > limit_d + tol)
        interfere_d = interfere_d + 1;
    end
    
    % 2. td (向下插管) 长度不能超过 h_真实 - safe_margin
    limit_td = h_rep_8 - safe_margin;
    if any(td_curr > limit_td + tol)
        interfere_td = interfere_td + 1;
    end
    
    % 3. tu (向上插管) 长度不能超过 h_真实 - safe_margin
    limit_tu = h_curr - safe_margin;
    if any(tu_curr > limit_tu + tol)
        interfere_tu = interfere_tu + 1;
    end
    
    % 4. Lu (水平插管) 长度不能超过 w_真实 - safe_margin
    limit_Lu = w_rep_4 - safe_margin;
    if any(Lu_curr > limit_Lu + tol)
        interfere_Lu = interfere_Lu + 1;
    end
end

% 打印穿模结果
fprintf('\n--- 空间干涉 (穿模) 诊断报告 ---\n');
if (interfere_d + interfere_td + interfere_tu + interfere_Lu) == 0
    fprintf('[完美通过] 0 个样本发生空间干涉！所有零件均安全放置于腔体内。\n');
else
    fprintf('[发现穿模] 检测到物理干涉！\n');
    fprintf('  - d 越界样本数: %d\n', interfere_d);
    fprintf('  - td 越界样本数: %d\n', interfere_td);
    fprintf('  - tu 越界样本数: %d\n', interfere_tu);
    fprintf('  - Lu 越界样本数: %d\n', interfere_Lu);
end

disp('======================================================');
disp('体检结束。');
disp('======================================================');