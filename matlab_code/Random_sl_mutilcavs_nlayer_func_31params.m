% Random_sl_mutilcavs_nlayer_func_31params.m (性能优化版)

function [slN_final] = Random_sl_mutilcavs_nlayer_func_31params(N, nlayer, ncell, nw, delta, H, W, lblim, ublim)
    
    disp('--- 正在使用【性能优化版】“31个浮动参数”函数生成数据 ---');
    
    lbw=lblim(1); ubw=ublim(1);
    lbd=lblim(5); ubd=ublim(5);
    lbt=lblim(3); ubt=ublim(3);
    lbtu=lblim(4); ubtu=ublim(4);
    lbLtot=lblim(6); ubLtot=ublim(6);
    
    tic;
    
    % --- 【核心修改】将所有能向量化的操作移出循环 ---
    
    % 1. 一次性生成所有浮动的 w (2*N 个参数)
    disp('  - 批量生成 w 参数...');
    Aeqw = ones(1, nw); 
    beqw = W - (nw + 1) * delta;
    lbwv = lbw * ones(nw, 1);
    ubwv = ubw * ones(nw, 1);
    w_variable = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, N); % 一次性生成N个样本

    % 2. 一次性生成所有浮动的 h, Lu, td, tu, Ltot
    disp('  - 批量生成 h, Lu, td, tu, Ltot 参数...');
    h_variable = 4.0 + (48.0 - 4.0) .* rand(ncell, N);
    Lu_variable = 10.0 + (30.0 - 10.0) .* rand(ncell, N);
    td_variable = lbt + (ubt - lbt) .* rand(nlayer * ncell, N);
    tu_variable = lbtu + (ubtu - lbtu) .* rand(ncell, N);
    Ltot_variable = lbLtot + (ubLtot - lbLtot) * rand(1, N);
    
    % 3. 循环生成依赖性参数 d (这部分仍需循环，但已大大减负)
    disp('  - 循环生成 d 参数 (此步仍需时间)...');
    d_variable = zeros(nlayer * ncell, N);
    nh_d = ncell / nw;
    for i = 1:N
        w_sample = w_variable(:, i);
        w_repeated_for_d = [w_sample(1)*ones(nh_d,1); w_sample(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        
        h_sample = h_variable(:, i);
        h_repeated_for_d = repmat(h_sample, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample = whmin_sample - 4 * delta;
        Lbd_sample = min(Lbd_sample, ubd);
        d_variable(:, i) = lbd + (Lbd_sample - lbd) .* rand(nlayer * ncell, 1);
    end

    % --- 拼接所有31个【随机】参数 ---
    disp('  - 正在拼接最终矩阵...');
    slN_final = [td_variable.', d_variable.', h_variable.', Lu_variable.', Ltot_variable.', w_variable.', tu_variable.'];
    
    toc;
    
    disp('31个全随机参数生成完毕。');
end