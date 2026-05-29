% target_objective_func.m
%
% 描述: 
%   V3 数据集的目标函数。
%   输入: 31个参数的向量 sl
%   输出: 一个“损失”值 (越低越好)

function [loss] = target_objective_func(sl)
    
    % --- 1. 定义我们想要的“精英”目标 ---
    % 目标 1: 宽带吸声"]
    BB_START_HZ = 500;
    BB_END_HZ = 2000;
    BB_THRESHOLD = 0.8;
    
    % 目标 2: 高 d 值
    D_TARGET_MIN = 15.0; % 惩罚那些 d 参数平均值低于 15 的解
    
    % --- 2. 运行物理仿真 ---
    % (设置仿真参数，必须与 generate_...m 一致)
    nlayer = 2; ncell = 4; nw = 2; delta = 0.7; mnum = 500; Norder = 9;
    fmin=60; fstep=1; fmax=3500; f0=fmin:fstep:fmax;
    
    [alpha, Z] = fun_structure_broadband_nlayer_HB_func(f0, sl, ncell, nw, delta, nlayer, mnum, Norder);
    
    % --- 3. 计算损失 ---
    
    % 损失 1: 宽带损失 (我们希望最小化 1.0 - Alpha)
    idx_start = find(f0 >= BB_START_HZ, 1, 'first');
    idx_end = find(f0 <= BB_END_HZ, 1, 'last');
    alpha_broadband = alpha(idx_start:idx_end);
    % 计算平均吸声与 0.9 (我们的目标) 的差距
    loss_bb = mean((0.9 - alpha_broadband).^2); 
    
    % 损失 2: d 参数不平衡损失 (我们希望 d 的平均值高)
    d_params = sl(9:16); % Params 8-15
    loss_d = max(0, D_TARGET_MIN - mean(d_params))^2; % 惩罚低 d 值

    % 最终总损失 (权重可调)
    loss = loss_bb + (loss_d * 0.1); 
end