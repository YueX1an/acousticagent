% Random_sl_mutilcavs_nlayer_func_31params_v3_elite.m
%
% 描述:
%   V3 "精英偏向" 生成器。
%   基于 "精英 DNA" 报告的发现。
%   【核心修改】：强制 h (Params 16-19) 在高值区 采样。

function [slN_final] = Random_sl_mutilcavs_nlayer_func_31params_v3_elite(N, nlayer, ncell, nw, delta, H, W, lblim, ublim)
    
    disp('  --- 正在使用【V3 精英偏向】生成器 (强制 h 在高值区) ---');
    
    lbw=lblim(1); ubw=ublim(1);
    lbd=lblim(5); ubd=ublim(5);
    lbt=lblim(3); ubt=ublim(3);
    lbtu=lblim(4); ubtu=ublim(4);
    lbLtot=lblim(6); ubLtot=ublim(6);
    
    % (V1 的代码 中没有 tic/toc，我们也保持一致)
    
    % 1. 一次性生成所有浮动的 w (2*N 个参数)
    % (保持全范围随机，以避免约束冲突)
    Aeqw = ones(1, nw); 
    beqw = W - (nw + 1) * delta;
    lbwv = lbw * ones(nw, 1);
    ubwv = ubw * ones(nw, 1);
    w_variable = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, N); 

    % 2. 一次性生成所有浮动的 Lu, td, tu, Ltot (保持全范围随机)
    Lu_variable = 10.0 + (30.0 - 10.0) .* rand(ncell, N);
    td_variable = lbt + (ubt - lbt) .* rand(nlayer * ncell, N);
    tu_variable = lbtu + (ubtu - lbtu) .* rand(ncell, N);
    Ltot_variable = lbLtot + (ubLtot - lbLtot) * rand(1, N);
    
    % --- 【V3 核心修改】 ---
    % 强制 h (Params 16-19) 在其物理范围的后半段 [26-48] 采样
    % (h 的物理范围是 4-48)
    h_min_elite = 26.0; %
    h_max_elite = 48.0;
    h_variable = h_min_elite + (h_max_elite - h_min_elite) .* rand(ncell, N);
    % --- 修改结束 ---
    
    % 3. 循环生成依赖性参数 d (此逻辑保持不变)
    % (d 将 *自然地* 倾向于高值，因为 h 是高值)
    d_variable = zeros(nlayer * ncell, N);
    nh_d = ncell / nw;
    for i = 1:N
        w_sample = w_variable(:, i);
        w_repeated_for_d = [w_sample(1)*ones(nh_d,1); w_sample(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        
        h_sample = h_variable(:, i);
        h_repeated_for_d = repmat(h_sample, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample = whmin_sample - 4 * delta; %
        Lbd_sample = min(Lbd_sample, ubd);
        Lbd_sample = max(Lbd_sample, lbd); % (增加保护，防止 Lbd_sample < lbd)
        
        d_variable(:, i) = lbd + (Lbd_sample - lbd) .* rand(nlayer * ncell, 1);
    end

    % --- 拼接所有31个【精英偏向】参数 ---
    slN_final = [td_variable.', d_variable.', h_variable.', Lu_variable.', Ltot_variable.', w_variable.', tu_variable.'];
    
end