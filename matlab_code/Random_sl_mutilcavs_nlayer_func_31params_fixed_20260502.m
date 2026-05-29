% Random_sl_mutilcavs_nlayer_func_31params_fixed.m
function [OutParams] = Random_sl_mutilcavs_nlayer_func_31params_fixed_20260502(N, nlayer, ncell, nw, delta, H, W, lblim, ublim)
    
    disp('--- 正在使用【Z轴防穿模版】生成器 (支持动态 Lu 和 tu 限制) ---');
    
    % 解析外部传入的上下限
    lbw = lblim(1);    ubw = ublim(1);
    lbLu = lblim(2);   ubLu = ublim(2);  
    lbt = lblim(3);    ubt = ublim(3);
    lbtu = lblim(4);   ubtu = ublim(4);
    lbd = lblim(5);    ubd = ublim(5);
    lbLtot = lblim(6); ubLtot = ublim(6);
    
    % ========================================================
    % 【关键修复区】：显式定义 h 和 w 的本地全局边界
    % 根据您的新 UI 界面，单层高度 h 的范围是 [10, 50]
    % ========================================================
    local_lbh = 10.0;  % <--- 修复点：补充了下限定义
    local_ubh = 50.0;  % 单层高度上限
    local_lbw = lbw;   % 主腔宽度下限
    local_ubw = ubw;   % 主腔宽度上限
    local_lbd = lbd;   % 孔径下限
    local_ubd = ubd;   % 孔径绝对上限
    
    % 初始化输出矩阵
    td_out   = zeros(N, nlayer * ncell);
    d_out    = zeros(N, nlayer * ncell);
    h_out    = zeros(N, ncell);
    Lu_out   = zeros(N, ncell);
    Ltot_out = zeros(N, 1);
    w_out    = zeros(N, nw);
    tu_out   = zeros(N, ncell);
    
    % w 参数约束矩阵 (w1 + w2 = W - 3*delta)
    Aeqw = ones(1, nw); 
    beqw = W - (nw + 1) * delta;
    
    % h 参数约束矩阵 (h1 + h2 = H - 3*delta)
    I_nw=eye(nw);
    nh=ncell/nw;
    one_nh=ones(nh,1);
    I_h=kron(I_nw,one_nh.');
    Aeqh=I_h;
    beqh=ones(nw,1)*(H-(nh+1)*delta); 
    
    nh_d = ncell / nw;
    tic;
    
    i = 1;
    rejected_count = 0;
    
    while i <= N
        % 1. 生成不受几何干涉影响的独立参数
        Ltot_final = lbLtot + (ubLtot - lbLtot) * rand(1, 1);
        td_final   = lbt + (ubt - lbt) .* rand(nlayer * ncell, 1);
        Lu_final   = lbLu + (ubLu - lbLu) .* rand(ncell, 1);
        
        % 2. 动态生成 tu (Z轴防穿模核心逻辑)
        tu_final = zeros(ncell, 1);
        is_z_rejected = false;
        for k = 1:ncell
            % 必须保证：Lu + tu + 至少3mm(上下底板及极小下腔) <= Ltot
            max_tu_allowed = Ltot_final - Lu_final(k) - 3.0; 
            actual_ubtu = min(ubtu, max_tu_allowed);
            
            % 如果随机出的 Lu 已经太大，导致连下限都塞不进去了，则废弃该样本
            if actual_ubtu < lbtu
                is_z_rejected = true;
                break;
            end
            tu_final(k) = lbtu + (actual_ubtu - lbtu) * rand(1);
        end
        
        if is_z_rejected
            rejected_count = rejected_count + 1;
            continue;
        end
        
        % 3. Hit and Run 生成空间耦合参数 (h 和 w)
        lbhv = local_lbh * ones(ncell, 1);
        ubhv = local_ubh * ones(ncell, 1);
        try
            h_final = NullSpace_HitandRun_func(Aeqh, beqh, [], [], lbhv, ubhv, 1);
        catch
            rejected_count = rejected_count + 1; continue; 
        end
        
        lbwv = local_lbw * ones(nw, 1);
        ubwv = local_ubw * ones(nw, 1);
        try
            w_final = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, 1);
        catch
            rejected_count = rejected_count + 1; continue; 
        end
        
        % 4. 计算 d 的物理上限并生成 (XY平面动态避障)
        w_repeated_for_d = [w_final(1)*ones(nh_d,1); w_final(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        h_repeated_for_d = repmat(h_final, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample_max = whmin_sample - 4 * delta; 
        
        actual_ubd = min(local_ubd, Lbd_sample_max);
        
        if any(actual_ubd < local_lbd)
            rejected_count = rejected_count + 1; continue; 
        end
        
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
        
        if mod(i, 5000) == 0
             fprintf('已生成 %d / %d 样本 (已拦截 %d 个越界样本)\n', i, N, rejected_count);
        end
        i = i + 1; 
    end
    
    OutParams.td   = td_out;
    OutParams.d    = d_out;
    OutParams.h    = h_out;
    OutParams.Lu   = Lu_out;
    OutParams.Ltot = Ltot_out;
    OutParams.w    = w_out;
    OutParams.tu   = tu_out;
    
    toc;
    fprintf('参数生成完毕。总计因物理防穿模拦截了 %d 个样本。\n', rejected_count);
end