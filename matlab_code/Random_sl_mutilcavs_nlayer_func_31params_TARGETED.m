% Random_sl_mutilcavs_nlayer_func_31params_TARGETED.m (约束冲突修正版)

function [slN_final] = Random_sl_mutilcavs_nlayer_func_31params_TARGETED(N, nlayer, ncell, nw, delta, H, W, lblim, ublim, hard_case_ratio)
    
    disp('--- 正在使用【靶向生成版 v2】函数生成数据 ---');
    fprintf('    - 困难样本生成比例: %.2f\n', hard_case_ratio);
    
    % 解析上下限
    lbw=lblim(1); ubw=ublim(1);
    lbd=lblim(5); ubd=ublim(5);
    lbt=lblim(3); ubt=ublim(3);
    lbtu=lblim(4); ubtu=ublim(4);
    lbLtot=lblim(6); ubLtot=ublim(6);
    
    % 初始化最终的参数矩阵
    slN_final = zeros(N, 31);
    
    tic;
    
    % --- 主循环：逐个生成样本 ---
    for i = 1:N
        % 首先，总是用同样的方式生成有约束的 w 参数
        Aeqw = ones(1, nw); 
        beqw = W - (nw + 1) * delta;
        lbwv = lbw * ones(nw, 1);
        ubwv = ubw * ones(nw, 1);
        w_final = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, 1);

        % 根据比例决定是生成困难样本还是普通样本
        if rand() < hard_case_ratio
            % --- 生成一个“困难样本” (只对无约束参数进行靶向) ---
            % 1. 高值困难区 (参数 0,1,2,3,24)
            % td_variable 的一部分 (0-3)
            td_hard_high = (lbt + ubt)/2 + rand(4, 1) * (ubt - lbt)/2;
            % td_variable 的其余部分保持随机
            td_hard_rest = lbt + rand(nlayer * ncell - 4, 1) * (ubt - lbt);
            td_final = [td_hard_high; td_hard_rest]; % 假设参数0-3是前4个
            
            Ltot_final = (lbLtot + ubLtot)/2 + rand(1, 1) * (ubLtot - lbLtot)/2;

            % 2. 低值困难区 (参数 16-23)
            h_final = 4.0 + rand(ncell, 1) * (48.0 - 4.0)/2;
            Lu_final = 10.0 + rand(ncell, 1) * (30.0 - 10.0)/2;
            
            % 3. 两端困难区 (参数 25, 26 -> 对应w) -- 我们在这里不做处理
            
            % 4. 中立参数
            tu_final = lbtu + rand(ncell, 1) * (ubtu - lbtu);

        else
            % --- 生成一个“普通随机样本”（使用原有逻辑）---
            h_final = 4.0 + (48.0 - 4.0) .* rand(ncell, 1);
            Lu_final = 10.0 + (30.0 - 10.0) .* rand(ncell, 1);
            td_final = lbt + (ubt - lbt) .* rand(nlayer * ncell, 1);
            tu_final = lbtu + (ubtu - lbtu) .* rand(ncell, 1);
            Ltot_final = lbLtot + (ubLtot - lbLtot) * rand(1, 1);
        end
        
        % 5. 生成依赖性参数 d (无论哪种情况都需要)
        nh_d = ncell / nw;
        w_repeated_for_d = [w_final(1)*ones(nh_d,1); w_final(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        
        h_repeated_for_d = repmat(h_final, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample = whmin_sample - 4 * delta;
        Lbd_sample = min(Lbd_sample, ubd);
        d_final = lbd + (Lbd_sample - lbd) .* rand(nlayer * ncell, 1);
        
        % --- 拼接单个样本的31个参数 ---
        sl_sample = [td_final.', d_final.', h_final.', Lu_final.', Ltot_final.', w_final.', tu_final.'];
        slN_final(i, :) = sl_sample;
    end
    
    toc;
    disp('31个靶向参数生成完毕。');
end