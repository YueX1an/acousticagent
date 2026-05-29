% =========================================================================
% Batch_Augment_Hologram_Seeds.m
% 面向全息图专属种子的批量安全微扰与并行扩增
% =========================================================================
clc; clear; close all;

% --- 1. 加载 Python 提取的批量祖先种子 ---
seeds = load('../hologram_targeted_seeds.txt');
N_seeds = size(seeds, 1);
fprintf('📥 成功加载全息专属种子库，共 %d 个极品祖先。\n', N_seeds);

% --- 2. 微调超参数 ---
num_children_per_seed = 50; % 每个祖先繁衍 50 个后代 (125 * 50 = 6250 个极品数据)
noise_ratio = 0.03;         % 保持 3% 的“手术刀级”微调，确保不出错
total_children = N_seeds * num_children_per_seed;

% 仿真参数
fmin = 50; fstep = 10; fmax = 2000;
f0 = fmin:fstep:fmax;
nlayer = 2; ncell = 4; nw = 2; delta = 1.0;
mnum = 500; Norder = 9;

fprintf('🚀 开始围绕 %d 个基准点进行 %.1f%% 振幅的微扰，预计生成 %d 个极品变体...\n', N_seeds, noise_ratio*100, total_children);

% --- 3. 批量生成变体矩阵 (应用动态边界保护) ---
valid_children_matrix = zeros(total_children, 31);

for i = 1:N_seeds
    best_sl = seeds(i, :);
    
    % 提取当前种子的隐式耦合约束 (极其关键的保护机制)
    sum_h12 = best_sl(17) + best_sl(18); 
    sum_h34 = best_sl(19) + best_sl(20); 
    sum_w12 = best_sl(26) + best_sl(27); 
    
    for j = 1:num_children_per_seed
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
        % 3. 层间接管 tu (28-31) - 无约束微调
        for p = 1:4
            fluct = best_sl(27+p) * noise_ratio * (2*rand() - 1);
            new_sl(27+p) = max(0.0, best_sl(27+p) + fluct); 
        end
        
        % 4. 高度 h (17-20) 零和微调
        dh_12 = (best_sl(17) * noise_ratio) * (2*rand() - 1);
        new_sl(17) = max(2.0, min(sum_h12 - 2.0, best_sl(17) + dh_12));
        new_sl(18) = sum_h12 - new_sl(17);
        
        dh_34 = (best_sl(19) * noise_ratio) * (2*rand() - 1);
        new_sl(19) = max(2.0, min(sum_h34 - 2.0, best_sl(19) + dh_34));
        new_sl(20) = sum_h34 - new_sl(19);
        
        % 5. 宽度 w (26-27) 零和微调
        dw = (best_sl(26) * noise_ratio) * (2*rand() - 1);
        new_sl(26) = max(2.0, min(sum_w12 - 2.0, best_sl(26) + dw));
        new_sl(27) = sum_w12 - new_sl(26);
        
        % 6. 安全孔径 d (9-16)
        for p = 1:8
            if p==1 || p==2 || p==5 || p==6; w_current = new_sl(26); else; w_current = new_sl(27); end
            if p==1 || p==5; h_current = new_sl(17); elseif p==2 || p==6; h_current = new_sl(18);
            elseif p==3 || p==7; h_current = new_sl(19); else; h_current = new_sl(20); end
            
            limit_d = max(0.5, min(w_current, h_current) - 1.0);
            fluct = best_sl(8+p) * noise_ratio * (2*rand() - 1);
            new_sl(8+p) = max(0.5, min(best_sl(8+p) + fluct, limit_d));
        end
        
        idx = (i-1)*num_children_per_seed + j;
        valid_children_matrix(idx, :) = new_sl;
    end
end
disp('✅ 变体生成完毕！启动多核并行声学仿真...');

% --- 4. 并行仿真获取真实声学响应 ---
all_alphas = zeros(total_children, length(f0));

tic;
parfor k = 1:total_children
    sl = valid_children_matrix(k, :);
    [alpha, ~] = fun_structure_broadband_nlayer_HB_func(f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
    all_alphas(k, :) = alpha';
end
toc;

% --- 5. 保存面向全息图的终极扩充数据集 ---
save('Hologram_Augmented_X.txt', 'valid_children_matrix', '-ascii');
save('Hologram_Augmented_Y.txt', 'all_alphas', '-ascii');

fprintf('🎉 突破成功！已将 %d 个面向全息图的专属高精度样本保存至 txt，可以拿去合并训练了！\n', total_children);