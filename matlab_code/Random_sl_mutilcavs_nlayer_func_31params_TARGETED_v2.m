% Random_sl_mutilcavs_nlayer_func_31params_TARGETED_v2.m (第二轮靶向生成版)

function [slN_final] = Random_sl_mutilcavs_nlayer_func_31params_TARGETED_v2(N, nlayer, ncell, nw, delta, H, W, lblim, ublim, hard_case_ratio)
    
    disp('--- 正在使用【第二轮靶向生成 v2】函数生成数据 ---');
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
        % 总是先用同样的方式生成有约束的 w 参数 (保持随机以避免约束冲突)
        Aeqw = ones(1, nw); 
        beqw = W - (nw + 1) * delta;
        lbwv = lbw * ones(nw, 1);
        ubwv = ubw * ones(nw, 1);
        % w_final 对应参数 25, 26
        w_final = NullSpace_HitandRun_func(Aeqw, beqw, [], [], lbwv, ubwv, 1); 

        if rand() < hard_case_ratio
            % --- 生成一个“困难样本” ---
            % disp(['Generating Hard Case Sample #', num2str(i)]); % 添加打印信息以便追踪

            % ==========================================================
            % --- 【请根据您的最新分析结果修改这里的参数索引和范围】---
            % 以下是基于我们“假设配方”的示例代码
            % ==========================================================
            
            % 1. 高值困难区 (参数 0, 1, 2, 3, 24)
            %    td_variable (索引 1-8 对应参数 0-7)
            td_final = zeros(nlayer * ncell, 1);
            high_value_indices_td = [1, 2, 3, 4]; % 假设参数0-3对应td的前4个索引
            for k = 1:(nlayer * ncell)
                if ismember(k, high_value_indices_td)
                    % 在后50%区间随机采样
                    td_final(k) = (lbt + ubt)/2 + rand() * (ubt - lbt)/2;
                else
                    % 其他td保持全范围随机
                    td_final(k) = lbt + rand() * (ubt - lbt);
                end
            end
            
            % Ltot_variable (索引 25 对应参数 24)
            Ltot_final = (lbLtot + ubLtot)/2 + rand() * (ubLtot - lbLtot)/2; % 在后50%区间随机采样

            % 2. 低值困难区 (参数 16, 17, ..., 23)
            %    h_variable (索引 17-20 对应参数 16-19)
            h_final = 4.0 + rand(ncell, 1) * (48.0 - 4.0)/2; % 在前半段区间随机采样
            %    Lu_variable (索引 21-24 对应参数 20-23)
            Lu_final = 10.0 + rand(ncell, 1) * (30.0 - 10.0)/2; % 在前半段区间随机采样

            % 3. 两端困难区 (参数 25, 26 -> 对应w) -- 【保持随机】
            % w_final 已经在循环外生成，保持全范围随机

            % 4. 中立参数 (例如 tu_variable, 对应参数 27-30)
            %    tu_variable (索引 28-31)
            tu_final = lbtu + rand(ncell, 1) * (ubtu - lbtu); % 保持全范围随机
            
            % --- 其他中立参数 (需要您根据完整配方补充) ---
            % 例如 d_variable (索引 9-16) 中不属于困难区的索引也应在此处生成
            % 示例：假设参数 8-15 均为中立
            % d_variable_neutral_part = ...; % 需要根据依赖关系计算d，见后文
            
            % ==========================================================
            % --- 【修改结束】 ---
            % ==========================================================

        else
            % --- 生成一个“普通随机样本” ---
            % disp(['Generating Normal Sample #', num2str(i)]); % 可选：取消注释以追踪
            h_final = 4.0 + (48.0 - 4.0) .* rand(ncell, 1);
            Lu_final = 10.0 + (30.0 - 10.0) .* rand(ncell, 1);
            td_final = lbt + (ubt - lbt) .* rand(nlayer * ncell, 1);
            tu_final = lbtu + (ubtu - lbtu) .* rand(ncell, 1);
            Ltot_final = lbLtot + (ubLtot - lbLtot) * rand(1, 1);
        end
        
        % 5. 生成依赖性参数 d (无论哪种情况都需要，且应保持随机)
        %    d_variable (索引 9-16 对应参数 8-15)
        nh_d = ncell / nw;
        w_repeated_for_d = [w_final(1)*ones(nh_d,1); w_final(2)*ones(nh_d,1)];
        w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1);
        h_repeated_for_d = repmat(h_final, nlayer, 1); % 使用已确定的h_final
        whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
        Lbd_sample = whmin_sample - 4 * delta;
        Lbd_sample = min(Lbd_sample, ubd); % 确保不超过全局上限
        Lbd_sample = max(Lbd_sample, lbd); % 确保不低于全局下限
        d_final = lbd + (Lbd_sample - lbd) .* rand(nlayer * ncell, 1); % 在计算出的可行范围内随机
        
        % --- 拼接单个样本的31个参数 ---
        sl_sample = [td_final.', d_final.', h_final.', Lu_final.', Ltot_final.', w_final.', tu_final.'];
        slN_final(i, :) = sl_sample;

        % 打印进度 (可选)
        if mod(i, 1000) == 0
             fprintf('Generated %d / %d samples...\n', i, N);
        end
    end
    
    toc;
    disp('第二轮靶向参数生成完毕。');
end