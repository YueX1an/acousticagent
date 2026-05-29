% find_perfect_absorber_ga.m
% 使用遗传算法 (GA) 寻找 60-2000Hz 宽频完美吸声的最优结构参数

clc; clear; close all;

disp('======================================================');
disp('启动遗传算法 (GA) 全局寻优：目标 -> 逼近宽频完美吸声');
disp('======================================================');

% --- 1. 物理常数与边界配置 ---
nw = 2; ncell = 4; nlayer = 2; delta = 0.7;
W_total = 49.4; H_total = 49.4;
nh = ncell / nw;

% 从你之前的设定提取上下限 (保持与生成数据集时一致)
lbw = 4.0;  ubw = 48.0;
lbt = 1.0;  ubt = 20.0;
lbtu = 0.0; ubtu = 20.0;
lbd = 1.0;  ubd = 30.0;
lbLu = 1.0; ubLu = 48.0;
lbLtot = 50.0; ubLtot = 200.0;

% 构建 31 维变量的全局 lb 和 ub
% 顺序: [td(8), d(8), h(4), Lu(4), Ltot(1), w(2), tu(4)]
lb = [lbt*ones(1,8), lbd*ones(1,8), 4.0*ones(1,4), lbLu*ones(1,4), lbLtot, lbw*ones(1,2), lbtu*ones(1,4)];
ub = [ubt*ones(1,8), ubd*ones(1,8), 48.0*ones(1,4), ubLu*ones(1,4), ubLtot, ubw*ones(1,2), ubtu*ones(1,4)];

% --- 2. 线性等式约束 (Aeq * x = beq) ---
% 保证 w 的总和与左右两列 h 的总和严格对齐
Aeq = zeros(3, 31);
beq = zeros(3, 1);

% Eq 1: w1 + w2 = 47.3 (w 的索引是 26, 27)
Aeq(1, 26:27) = 1;
beq(1) = W_total - (nw + 1) * delta; 

% Eq 2: h1 + h2 = 47.3 (左列, 索引 17, 18)
Aeq(2, 17:18) = 1;
beq(2) = H_total - (nh + 1) * delta;

% Eq 3: h3 + h4 = 47.3 (右列, 索引 19, 20)
Aeq(3, 19:20) = 1;
beq(3) = H_total - (nh + 1) * delta;

% --- 3. 遗传算法配置项 (防止早熟躺平版) ---
options = optimoptions('ga', ...
    'PopulationSize', 100, ...          % 【调大】将种群数量从 50 扩大到 100，增加基因多样性
    'MaxGenerations', 50, ...           % 最大进化代数
    'MaxStallGenerations', 20, ...      % 【核心】连续 20 代找不到更好的解，才允许它停止！(防早退)
    'Display', 'iter', ...
    'UseParallel', true, ...
    'ConstraintTolerance', 1e-3, ...
    'FunctionTolerance', 1e-6);         % 【核心】将停止阈值调到极小，逼迫它在微小变化中继续压榨极限

% --- 4. 启动优化求解 ---
disp('正在配置并行池并启动进化，这将耗费较长时间，请耐心等待...');
tic;
% 调用 GA: [x, fval] = ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options)
[best_slN, best_cost, exitflag, output] = ga(@objective_function, 31, [], [], Aeq, beq, lb, ub, @physical_constraints, options);
toc;

disp('======================================================');
fprintf('优化完成！最优得分 (Cost): %.4f\n', best_cost);
disp('最优结构参数 slN 为：');
disp(best_slN);

% --- 5. 验证并绘制最优解 ---
fmin = 60; fstep = 1; fmax = 3500;
f0 = fmin:fstep:fmax;
disp('正在计算最优解的全频段吸收曲线...');
[alpha_best, ~] = fun_structure_broadband_nlayer_HB_func(f0, best_slN, ncell, nw, delta, nlayer, 500, 9);

figure;
plot(f0, alpha_best, 'r-', 'LineWidth', 2);
xlim([60 2000]); ylim([0 1.05]);
grid on;
xlabel('Frequency (Hz)'); ylabel('Absorption Coefficient');
title(sprintf('GA 寻优结果 (50-2000Hz 平均吸收率: %.2f%%)', mean(alpha_best(f0<=2000))*100));


% =========================================================================
% 内部函数 1：目标评估函数 (Cost Function)
% =========================================================================
function cost = objective_function(x)
    % 提取外部参数以供仿真
    ncell = 4; nw = 2; delta = 0.7; nlayer = 2; mnum = 500; Norder = 9;
    
    % --- 🚀 提速核心修改区 ---
    % 1. 将步长 fstep 改为 10Hz 甚至 20Hz (寻优不需要 1Hz 那么精细)
    % 2. 将 fmax 直接卡在 2000Hz (不需要计算 3500Hz 浪费算力)
    fmin = 60; fstep = 20; fmax = 2000; 
    f0 = fmin:fstep:fmax;
    
    % 调用正向仿真函数
    try
        % 注意：这里计算的数据点从原本的 3441 个骤降到了 98 个，速度直接起飞！
        [alpha_a, ~] = fun_structure_broadband_nlayer_HB_func(f0, x, ncell, nw, delta, nlayer, mnum, Norder);
        
        % 目标：让平均吸收率尽可能接近 1
        cost = 1 - mean(alpha_a);
        
        % 附加惩罚：惩罚曲线中的深谷，让它尽量平坦 (可选，推荐加上)
        cost = cost + 0.5 * (1 - min(alpha_a)); 
        
    catch
        % 如果遇到矩阵奇异等严重报错导致崩溃，给予极大惩罚
        cost = 1000; 
    end
end

% =========================================================================
% 内部函数 2：非线性物理约束 (防穿模保护)
% =========================================================================
function [c, ceq] = physical_constraints(x)
    % 解析数组
    td_curr = x(1:8);
    d_curr  = x(9:16);
    h_curr  = x(17:20);
    Lu_curr = x(21:24);
    w_curr  = x(26:27);
    tu_curr = x(28:31);
    
    delta = 0.7;
    safe_margin = 1.5 * delta;
    
    % 重构映射关系
    w_rep_4 = [w_curr(1), w_curr(1), w_curr(2), w_curr(2)];
    w_rep_8 = [w_rep_4, w_rep_4];
    h_rep_8 = [h_curr, h_curr];
    
    % 计算不等式约束 c (要求 c <= 0 代表合法)
    % 1. d 不能超过 min(w,h) - 4*delta => d - limit <= 0
    limit_d = min(w_rep_8, h_rep_8) - 4 * delta;
    c_d = d_curr - limit_d;
    
    % 2. td 长度不能超过 h_真实 - safe_margin
    limit_td = h_rep_8 - safe_margin;
    c_td = td_curr - limit_td;
    
    % 3. tu 长度不能超过 h_真实 - safe_margin
    limit_tu = h_curr - safe_margin;
    c_tu = tu_curr - limit_tu;
    
    % 4. Lu 长度不能超过 w_真实 - safe_margin
    limit_Lu = w_rep_4 - safe_margin;
    c_Lu = Lu_curr - limit_Lu;
    
    % 汇总所有的不等式约束
    c = [c_d, c_td, c_tu, c_Lu];
    
    % 非线性等式约束留空 (已在 Aeq 中定义)
    ceq = [];
end