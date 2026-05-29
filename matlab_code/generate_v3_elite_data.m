% generate_v3_elite_data.m
%
% 描述: 
%   使用遗传算法 (ga) 和真实物理仿真，
%   主动搜索并生成一个 V3 "精英" 数据集。

clc; clear;

% --- 1. 配置 ---
N = 10; % 目标：生成 5 万个“精英”样本 (这会非常慢！)
n_params = 31;

% 定义 31 个参数的边界 (Lb = Lower, Ub = Upper)
% (td: 0-7, d: 8-15, h: 16-19, Lu: 20-23, Ltot: 24, w: 25-26, tu: 27-30)
Lb = [ 1, 1, 1, 1, 1, 1, 1, 1, ... % td
       1, 1, 1, 1, 1, 1, 1, 1, ... % d
       4, 4, 4, 4, ...             % h
       10, 10, 10, 10, ...         % Lu
       50, ...                     % Ltot
       4, 4, ...                   % w
       0, 0, 0, 0 ];              % tu
Ub = [ 20, 20, 20, 20, 20, 20, 20, 20, ... % td
       30, 30, 30, 30, 30, 30, 30, 30, ... % d
       48, 48, 48, 48, ...             % h
       30, 30, 30, 30, ...             % Lu
       200, ...                    % Ltot
       48, 48, ...                   % w
       20, 20, 20, 20 ];              % tu

% --- 2. 设置 GA 选项 ---
% (使用并行计算)
options = optimoptions('ga', ...
    'Display', 'iter', ...
    'PopulationSize', 100, ... % 种群大小
    'MaxGenerations', 50, ...  % 进化代数
    'UseParallel', true);      % <-- 关键：使用并行池

% --- 3. 运行 GA 搜索 N 次 ---
slN_v3 = zeros(N, n_params);
fL = length(60:1:3500);
alphaM_v3 = zeros(fL, N);
RZM_v3 = zeros(fL, N);
XZM_v3 = zeros(fL, N);

disp(['--- 开始 V3 精英数据生成 (N=', num2str(N), ') ---']);
disp('这会非常非常慢，每次搜索都需要几十分钟。');
tic;

% (我们不能并行 N 次 GA，因为 GA 内部已经并行了)
% (我们将串行 N 次 GA)
for i = 1:N
    fprintf('\n--- 正在搜索精英样本 %d / %d ---\n', i, N);
    
    [X_found, fval] = ga(@target_objective_func, ...
        n_params, ...
        [], [], [], [], ... % A, b, Aeq, beq (我们用非线性约束)
        Lb, Ub, ...
        @physical_constraints, ... % <-- 我们的硬约束
        options);
    
    fprintf('  > 样本 %d 找到！ 最佳损失: %f\n', i, fval);
    
    % 4. 保存找到的参数
    slN_v3(i, :) = X_found;
    
    % 5. 重新运行一次仿真以获取数据 (或从 ga 的 fval 中获取)
    [alpha, Z] = fun_structure_broadband_nlayer_HB_func(60:1:3500, X_found, ncell, nw, delta, nlayer, mnum, Norder);
    alphaM_v3(:, i) = alpha;
    RZM_v3(:, i) = real(Z);
    XZM_v3(:, i) = imag(Z);
end
toc;

% --- 6. 保存 V3 数据集 ---
disp('--- 正在保存 V3 精英数据集 ---');
save('v3_elite_slN.txt', 'slN_v3', '-ascii');
save('v3_elite_alphaM.txt', 'alphaM_v3', '-ascii');
save('v3_elite_RZM.txt', 'RZM_v3', '-ascii');
save('v3_elite_XZM.txt', 'XZM_v3', '-ascii');
disp('V3 精英数据集保存完成！');