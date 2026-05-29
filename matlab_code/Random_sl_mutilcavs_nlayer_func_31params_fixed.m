% Random_sl_mutilcavs_nlayer_func_31params_fixed.m
% 纯正常生成版：剔除强制分箱，严格遵守物理空间限制
function [OutParams] = Random_sl_mutilcavs_nlayer_func_31params_fixed(N, nlayer, ncell, nw, delta, H, W, lblim, ublim)
    
    disp('--- 正在使用【正常生成版】函数 (严格遵守物理空间限制) ---');
    
    % 解析上下限
    lbw = lblim(1);    ubw = ublim(1);
    lbLu = lblim(2);   ubLu = ublim(2);  
    lbt = lblim(3);    ubt = ublim(3);
    lbtu = lblim(4);   ubtu = ublim(4);
    lbd = lblim(5);    ubd = ublim(5);
    lbLtot = lblim(6); ubLtot = ublim(6);
    
    % 初始化输出矩阵 (N 个样本)
    td_out   = zeros(N, nlayer * ncell);
    d_out    = zeros(N, nlayer * ncell);
    h_out    = zeros(N, ncell);
    Lu_out   = zeros(N, ncell);
    Ltot_out = zeros(N, 1);
    w_out    = zeros(N, nw);
    tu_out   = zeros(N, ncell);
    
    % w 参数约束
    Aeqw = ones(1, nw); 
    beqw = W - (nw + 1) * delta;
    
    % h 参数约束 
    I_nw=eye(nw);
    nh=ncell/nw;
    one_nh=ones(nh,1);
    I_h=kron(I_nw,one_nh.');
    Aeqh=I_h;
    beqh=ones(nw,1)*(H-(nh+1)*delta); 
    
    % d 参数的依赖性
    nh_d = ncell / nw;
    
    tic;
    
    % --- 主循环：逐个生成样本 ---
    i = 1;
    rejected_count = 0;
    
    % 正常生成的全局边界，不再动态修改
    local_lbd = lbd; 
    local_ubd = ubd; 
        
    local_ubh = 48.0; 
    local_lbw = lbw; 
    local_ubw = ubw; 
    
    while i <= N
        % 1. 生成独立参数
        Ltot_final = lbLtot + (ubLtot - lbLtot) * rand(1, 1);
        td_final   = lbt + (ubt - lbt) .* rand(nlayer * ncell, 1);
        tu_final   = lbtu + (ubtu - lbtu) .* rand(ncell, 1);
        Lu_final   = lbLu + (ubLu - lbLu) .* rand(ncell, 1);
        
        % 独立参数 (h)
        lbhv = local_lbh * ones(ncell, 1);
        ubhv = local_ubh * ones(ncell, 1);
        try
            h_final = NullSpace_HitandRun_func(Aeqh, beqh, [], [], lbhv, ubhv, 1);
        catch
            rejected_count = rejected_count + 1;
            continue; 
        end
        
        % 独立参数 (w)
        lbwv = local_lbw * ones(nw, 1);
        ubwv = local_ubw * ones(nw, 1);
        try
            w_final = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, 1);
        catch
            rejected_count = rejected_count + 1;
            continue; 
        end
        
        % 2. 计算 d 的物理上限 (核心动态安全范围)
        w_repeated_for_d = [w_final(1)*ones(nh_d,1); w_final(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        h_repeated_for_d = repmat(h_final, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample_max = whmin_sample - 4 * delta; 
        
        % 取设定的 ubd 与实际物理允许的最大值的较小者
        actual_ubd = min(local_ubd, Lbd_sample_max);
        
        % 3. 拒绝检查
        if any(actual_ubd < local_lbd)
            % 如果物理允许的上限比我们要求的下限 (1.0) 还要小，说明空间不够，换一组 h 和 w
            rejected_count = rejected_count + 1;
            continue; 
        end
        
        % 4. 100% 安全生成 d 参数
        d_final = zeros(nlayer * ncell, 1);
        for k = 1:(nlayer * ncell)
            d_final(k) = local_lbd + (actual_ubd(k) - local_lbd) * rand(1);
        end
        
        % 5. 存储参数
        td_out(i, :)   = td_final.';
        d_out(i, :)    = d_final.';
        h_out(i, :)    = h_final.';
        Lu_out(i, :)   = Lu_final.';
        Ltot_out(i, :) = Ltot_final.';
        w_out(i, :)    = w_final.';
        tu_out(i, :)   = tu_final.';
        
        % 打印进度
        if mod(i, 5000) == 0
             fprintf('已生成 %d / %d 样本 (已拒绝 %d 个空间过小的样本)\n', i, N, rejected_count);
        end
        
        i = i + 1; 
    end
    
    % 将结果封装为结构体输出
    OutParams.td   = td_out;
    OutParams.d    = d_out;
    OutParams.h    = h_out;
    OutParams.Lu   = Lu_out;
    OutParams.Ltot = Ltot_out;
    OutParams.w    = w_out;
    OutParams.tu   = tu_out;
    
    toc;
    fprintf('参数生成完毕。总计因物理干涉拒绝了 %d 个样本。\n', rejected_count);
end