% =========================================================================
% FineTune_Best_Seed.m
% 基于单一黄金基准参数的局部流形精细微调与寻优 (安全保护版)
% =========================================================================
clc; clear; close all;

% --- 1. 载入黄金基准点 (你的 31 维参数) ---
best_sl = [3.0000 1.0000 1.0505 1.0000 1.9500 5.4500 17.8490 5.9611 ...
           8.4500 6.5750 12.0864 16.5768 4.2197 9.1565 13.4500 16.5768 ...
           21.7947 25.5053 27.5232 19.7768 44.0000 26.2500 36.0012 49.9869 ...
           57.0000 27.5232 19.7768 11.5000 5.5000 29.5505 13.1875];

% 目标设定
target_vector = [0.4, 0.8, 0.8, 0.8];
target_freqs = [400, 900, 1400, 1900];
fmin = 50; fstep = 10; fmax = 2000;
f0 = fmin:fstep:fmax;
target_idx = (target_freqs - fmin) / fstep + 1; 

% 仿真参数
nlayer = 2; ncell = 4; nw = 2; delta = 1.0;
mnum = 500; Norder = 9;

% --- 2. 动态提取基准点的隐式物理约束 ---
sum_h12 = best_sl(17) + best_sl(18); % 约 47.3
sum_h34 = best_sl(19) + best_sl(20); % 约 47.3
sum_w12 = best_sl(26) + best_sl(27); % 约 47.3
% Ltot (best_sl(25)) 维持 57.0 不变

% --- 3. 微调超参数 ---
num_children = 5000;    % 生成 5000 个精细变体
noise_ratio = 0.03;     % 🔴 手术刀级微调：只允许 3% 的波动！

fprintf('成功加载黄金基准点，动态约束提取完毕 (总高宽耦合值: %.2f)。\n', sum_h12);
fprintf('开始围绕基准点进行 %.1f%% 振幅的精细微扰，生成 %d 个变体...\n', noise_ratio*100, num_children);

% --- 4. 生成变体矩阵 (安全微扰模式) ---
valid_children_matrix = zeros(num_children, 31);

for j = 1:num_children
    new_sl = best_sl;
    
    % 1. 管长 t (1-8)
    for p = 1:8
        fluct = best_sl(p) * noise_ratio * (2*rand() - 1);
        new_sl(p) = max(1.0, min(200.0, best_sl(p) + fluct));
    end
    
    % 2. 上层腔长 Lu (21-24)
    for p = 21:24
        fluct = best_sl(p) * noise_ratio * (2*rand() - 1);
        new_sl(p) = max(10.0, min(50.0, best_sl(p) + fluct));
    end
    
    % 3. 层间接管 tu (28-31)
    % (⚠️ 取消针对 Ltot 的强制截断，完全尊重黄金种子的原始比例)
    for p = 1:4
        fluct = best_sl(27+p) * noise_ratio * (2*rand() - 1);
        new_sl(27+p) = max(0.0, best_sl(27+p) + fluct); 
    end
    
    % 4. 高度 h (17-20) 零和微调
    dh_12 = (best_sl(17) * noise_ratio) * (2*rand() - 1);
    h1_cand = best_sl(17) + dh_12;
    new_sl(17) = max(2.0, min(sum_h12 - 2.0, h1_cand));
    new_sl(18) = sum_h12 - new_sl(17);
    
    dh_34 = (best_sl(19) * noise_ratio) * (2*rand() - 1);
    h3_cand = best_sl(19) + dh_34;
    new_sl(19) = max(2.0, min(sum_h34 - 2.0, h3_cand));
    new_sl(20) = sum_h34 - new_sl(19);
    
    % 5. 宽度 w (26-27) 零和微调
    dw = (best_sl(26) * noise_ratio) * (2*rand() - 1);
    w1_cand = best_sl(26) + dw;
    new_sl(26) = max(2.0, min(sum_w12 - 2.0, w1_cand));
    new_sl(27) = sum_w12 - new_sl(26);
    
    % 6. 安全孔径 d (9-16)
    for p = 1:8
        if p==1 || p==2 || p==5 || p==6; w_current = new_sl(26); else; w_current = new_sl(27); end
        if p==1 || p==5; h_current = new_sl(17); elseif p==2 || p==6; h_current = new_sl(18);
        elseif p==3 || p==7; h_current = new_sl(19); else; h_current = new_sl(20); end
        
        limit_d = max(0.5, min(w_current, h_current) - 1.0);
        fluct = best_sl(8+p) * noise_ratio * (2*rand() - 1);
        d_candidate = best_sl(8+p) + fluct;
        new_sl(8+p) = max(0.5, min(d_candidate, limit_d));
    end
    
    % (⚠️ 已删除错误的 new_sl(25) = sum(...) 累加代码)
    % new_sl(25) 会自然保留 best_sl(25) 的 57.0 不动
    
    valid_children_matrix(j, :) = new_sl;
end

disp('变体生成完毕！启动并行仿真评估...');

% --- 5. 并行仿真与单边松弛误差评估 ---
all_mses = zeros(num_children, 1);
all_alphas = zeros(num_children, length(f0));

tic;
parfor k = 1:num_children
    sl = valid_children_matrix(k, :);
    [alpha, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
    
    current_focus = alpha(target_idx)';
    
    % 引入单边松弛误差 (Asymmetric Loss)
    errors = zeros(1, 4);
    for v = 1:4
        if target_vector(v) >= 0.8 && current_focus(v) > target_vector(v)
            errors(v) = 0; % 超额完成高吸收，不计入误差
        else
            errors(v) = (current_focus(v) - target_vector(v))^2;
        end
    end
    
    all_mses(k) = mean(errors);
    all_alphas(k, :) = alpha';
end
toc;

% --- 6. 排行榜与结果保存 ---
[sorted_mses, sort_idx] = sort(all_mses, 'ascend');

fprintf('\n=== 🏆 局部微调评估报告 ===\n');
fprintf('最佳变体的 MSE 达到了: %.6f\n', sorted_mses(1));

% 提取前 100 名极品参数用于你的神经网络训练
top_k = min(100, num_children);
elite_params = valid_children_matrix(sort_idx(1:top_k), :);
elite_alphas = all_alphas(sort_idx(1:top_k), :);

save('finetuned_elite_params.txt', 'elite_params', '-ascii');
save('finetuned_elite_alphas.txt', 'elite_alphas', '-ascii');

fprintf('已将前 %d 名极品结构保存至 finetuned_elite_params.txt，可以直接合并到训练集！\n', top_k);