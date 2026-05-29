% physical_constraints.m
%
% 描述: 
%   为 MATLAB 的 'ga' 优化器定义 *所有* 物理约束
%   sl 是 31 个参数的向量
%   【已修正】修复了 h_repeated_for_d 的维度不兼容错误

function [c, ceq] = physical_constraints(sl)
    
    % --- 1. 提取所需参数 ---
    d_params = sl(9:16); % Params 8-15
    h_params = sl(17:20); % Params 16-19 (是一个 [1, 4] 的行向量)
    w_params = sl(26:27); % Params 25-26 (是一个 [1, 2] 的行向量)
    
    % --- 2. 约束 1: w 的总和 ---
    nlayer = 2; ncell = 4; nw = 2; delta = 0.7; W = 49.4;
    target_w_sum = W - (nw + 1) * delta; % 应该等于 47.3
    
    % ceq (等式约束): 必须为 0
    ceq = sum(w_params) - target_w_sum;

    % --- 3. 约束 2: d 的依赖 ---
    % d(i) <= min(w_i, h_i) - 4*delta
    nh_d = ncell / nw; % 4 / 2 = 2
    
    % --- 修正 w_repeated_for_d 的生成 ---
    % w_params 是 (1, 2)
    w_repeated_for_d = [w_params(1)*ones(nh_d,1); w_params(2)*ones(nh_d,1)]; % (4, 1)
    w_repeated_for_d = repmat(w_repeated_for_d, nlayer, 1); % (8, 1)
    
    % --- 【核心修正】修正 h_repeated_for_d 的生成 ---
    % h_params 是 (1, 4)
    h_params_col = h_params.'; % 将其转置为 (4, 1)
    h_repeated_for_d = repmat(h_params_col, nlayer, 1); % (8, 1)
    % --- 修正结束 ---
    
    % (现在 w_repeated_for_d 和 h_repeated_for_d 都是 (8, 1)，维度兼容)
    whmin_sample = min(w_repeated_for_d, h_repeated_for_d);
    d_max_allowed = whmin_sample - 4 * delta; % (8, 1)
    
    % c (不等式约束): 必须 <= 0
    % d_params 是 (1, 8)，需要转置
    c = d_params.' - d_max_allowed; % (8, 1)
    
end