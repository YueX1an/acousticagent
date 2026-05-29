% =========================================================================
% Hunt_16_States.m
% 为 4 频点复用专门定制的 16 态基础基因库寻优算法
% =========================================================================
clc; clear; close all;

% --- 1. 物理环境与目标频率设定 ---
f0 = 50:10:2000;
freq_targets = [400, 900, 1400, 1900];
% 找到目标频率在 f0 中的索引
[~, idx1] = min(abs(f0 - freq_targets(1)));
[~, idx2] = min(abs(f0 - freq_targets(2)));
[~, idx3] = min(abs(f0 - freq_targets(3)));
[~, idx4] = min(abs(f0 - freq_targets(4)));
target_idx = [idx1, idx2, idx3, idx4];

nlayer = 2; ncell = 4; nw = 2; delta = 1.0; mnum = 500; Norder = 9;

% --- 2. 生成 16 种逻辑状态矩阵 ---
% 0 代表要求低吸收 (0.4)，1 代表要求高吸收 (0.8)
states_bin = dec2bin(0:15) - '0'; 
target_alphas = states_bin * 0.4 + 0.4; % 将 0和1 映射为 0.4 和 0.8

% 预分配空间存储 16 个精英参数
elite_16_params = zeros(16, 31);
elite_16_errors = zeros(16, 1);

% --- 3. GA 优化边界 (加入 Z 轴防穿模的线性约束) ---
lb = [ones(1,8)*1,  ones(1,8)*1.0, ones(1,4)*10, ones(1,4)*15, 57, ones(1,2)*15, zeros(1,4)];
ub = [ones(1,8)*200,ones(1,8)*30,  ones(1,4)*50, ones(1,4)*40, 57, ones(1,2)*40, ones(1,4)*200];

% 面积约束 (h1+h2=46.7, h3+h4=46.7, w1+w2=46.7)
Aeq = zeros(3, 31); beq = zeros(3, 1);
Aeq(1, 17) = 1; Aeq(1, 18) = 1; beq(1) = 46.7; 
Aeq(2, 19) = 1; Aeq(2, 20) = 1; beq(2) = 46.7;
Aeq(3, 26) = 1; Aeq(3, 27) = 1; beq(3) = 46.7;

% --- 4. 开始逐个攻克 16 种状态 ---
options = optimoptions('ga', 'PopulationSize', 100, 'MaxGenerations', 50, 'Display', 'iter');

for i = 1:16
    fprintf('\n🚀 正在攻克第 %d/16 种状态: 目标吸收率 [%.1f, %.1f, %.1f, %.1f]\n', ...
        i, target_alphas(i,1), target_alphas(i,2), target_alphas(i,3), target_alphas(i,4));
    
    current_target = target_alphas(i, :);
    
    % 定义适应度函数 (MSE + 串扰惩罚)
    fitness_fcn = @(x) obj_func_multiplexing(x, current_target, target_idx, f0, ncell, nw, delta, nlayer, mnum, Norder);
    
    % 运行 GA 寻找这一个状态的最优解 (这里可以加上您之前写的非线性防穿模约束)
    [opt_x, fval] = ga(fitness_fcn, 31, [], [], Aeq, beq, lb, ub, [], options);
    
    elite_16_params(i, :) = opt_x;
    elite_16_errors(i) = fval;
end

% --- 5. 保存结果 ---
save("elite_16_dictionary.txt", "elite_16_params", "-ascii");
disp('🎉 16 态基础字典生成完毕！');

% =========================================================================
% 适应度函数：带串扰惩罚的评估
% =========================================================================
function cost = obj_func_multiplexing(x, target_vals, target_idx, f0, ncell, nw, delta, nlayer, mnum, Norder)
    % 1. 物理穿模极速检查 (如果穿模，直接返回巨大惩罚值)
    % (此处可贴入之前写的 Z轴厚度和孔径检查逻辑，若不合法返回 1000)
    
    % 2. 计算光谱
    alpha = fun_structure_broadband_nlayer_HB_func(f0, x, ncell, nw, delta, nlayer, mnum, Norder);
    
    % 3. 提取 4 个频点的值
    actual_vals = alpha(target_idx).';
    
    % 4. 计算 MSE
    mse_cost = sum((actual_vals - target_vals).^2);
    
    % 5. 🌟 串扰惩罚 (对于要求为 0.4 的频点，如果实际算出来偏高，给予双倍惩罚)
    penalty = 0;
    for k = 1:4
        if target_vals(k) == 0.4 && actual_vals(k) > 0.5
            penalty = penalty + (actual_vals(k) - 0.4)^2 * 5.0; 
        end
    end
    
    cost = mse_cost + penalty;
end