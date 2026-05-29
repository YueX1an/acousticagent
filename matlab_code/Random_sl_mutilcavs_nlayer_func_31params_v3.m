% Random_sl_mutilcavs_nlayer_func_31params_v3.m (V3 均衡生成版)
% 目标：解决 d 参数 (Params 8-15) 的严重不平衡问题

function [slN_final] = Random_sl_mutilcavs_nlayer_func_31params_v3(N, nlayer, ncell, nw, delta, H, W, lblim, ublim)
    
    disp('--- 正在使用【V3 均衡生成】函数 (解决 d 参数不平衡) ---');
    
    % 解析上下限
    lbw=lblim(1); ubw=ublim(1);
    lbd=lblim(5); ubd=ublim(5);
    lbt=lblim(3); ubt=ublim(3);
    lbtu=lblim(4); ubtu=ublim(4);
    lbLtot=lblim(6); ubLtot=ublim(6);
    
    % 初始化最终的参数矩阵
    slN_final = zeros(N, 31);
    
    % w 参数约束
    Aeqw = ones(1, nw); 
    beqw = W - (nw + 1) * delta;
    
    % d 参数的依赖性
    nh_d = ncell / nw;
    
    tic;
    
    % --- 主循环：逐个生成样本 ---
    i = 1;
    rejected_count = 0;
    fprintf('开始生成 %d 个样本...\n', N);
    
    while i <= N
        % --- 1. 决定当前要生成的“d参数箱” ---
        target_bin = mod(i-1, 3); % 0, 1, 2 (对应 Low, Mid, High)
        
        local_lbd = lbd; % 局部d下限
        local_ubd = ubd; % 局部d上限
        local_lbh = 4.0; % 局部h下限
        local_ubh = 48.0; % 局部h上限
        local_lbw = lbw; % 局部w下限
        local_ubw = ubw; % 局部w上限

        % 物理约束: d_max <= min(w,h) - 4*delta (delta=0.7, 4*delta=2.8)
        
        if target_bin == 0 % 目标: d 在 [1, 10] (Low bin)
            local_ubd = 10.0;
            % h 和 w 可以全范围随机
            
        elseif target_bin == 1 % 目标: d 在 [10, 20] (Mid bin)
            local_lbd = 10.0;
            local_ubd = 20.0;
            % d 要达到 20, min(w,h) 必须 >= 20 + 2.8 = 22.8
            local_lbh = 22.8; 
            local_lbw = 22.8;
            
        elseif target_bin == 2 % 目标: d 在 [20, 30] (High bin)
            local_lbd = 20.0;
            local_ubd = 30.0;
            % d 要达到 30, min(w,h) 必须 >= 30 + 2.8 = 32.8
            local_lbh = 32.8;
            local_lbw = 32.8;
        end
        
        % --- 2. 生成独立参数 ---
        
        % 独立参数 (Ltot, td, tu) 保持 V1 的完美平衡随机采样
        Ltot_final = lbLtot + (ubLtot - lbLtot) * rand(1, 1);
        td_final = lbt + (ubt - lbt) .* rand(nlayer * ncell, 1);
        tu_final = lbtu + (ubtu - lbtu) .* rand(ncell, 1);
        
        % 独立参数 (h) 根据目标箱的下限生成
        h_final = local_lbh + (local_ubh - local_lbh) .* rand(ncell, 1);
        
        % 独立参数 (w) 根据目标箱的下限生成
        lbwv = local_lbw * ones(nw, 1);
        ubwv = local_ubw * ones(nw, 1);
        % 【注意】NullSpace_HitandRun_func 可能会因为约束太强而失败
        try
            w_final = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, 1);
        catch
            % 如果 NullSpace 失败 (例如 [32.8, 48] 范围太窄无法满足总和)
            rejected_count = rejected_count + 1;
            continue; % 拒绝此样本，重新循环
        end
        
        % --- 3. 生成依赖参数 (d) ---
        d_final = local_lbd + (local_ubd - local_lbd) .* rand(nlayer * ncell, 1);
        
        % --- 4. 【关键】拒绝采样：检查 d 是否满足物理约束 ---
        w_repeated_for_d = [w_final(1)*ones(nh_d,1); w_final(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        h_repeated_for_d = repmat(h_final, nlayer, 1);
        
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample_max = whmin_sample - 4 * delta; % 计算 d 的真实物理上限
        
        if any(d_final > Lbd_sample_max)
            % 拒绝！d 的采样值超过了 w,h 决定的物理上限
            rejected_count = rejected_count + 1;
            continue; % 重新循环，i 不增加
        end

        % --- 5. 拼接样本 (如果所有检查都通过) ---
        sl_sample = [td_final.', d_final.', h_final.', Lu_final.', Ltot_final.', w_final.', tu_final.'];
        slN_final(i, :) = sl_sample;
        
        % 打印进度
        if mod(i, 1000) == 0
             fprintf('V3 已生成 %d / %d 样本 (已拒绝 %d 个无效样本)\n', i, N, rejected_count);
        end
        
        i = i + 1; % 只有成功后才增加 i
    end
    
    toc;
    fprintf('V3 均衡参数生成完毕。总计拒绝了 %d 个无效样本。\n', rejected_count);
end