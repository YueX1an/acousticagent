% --- 1. 定义初始猜测参数 (可以放入您认为吸声好但越界的参数) ---
% 这里放入的是您提供的 sl_opt，算法会以此为起点进行寻找
x0 = [1.0000 14.0000 1.0000 2.0000 19.0000 19.0000 2.0000 1.0000 ...
      9.4500 11.2000 10.0571 10.3905 9.4500 3.2000 3.2000 12.1905 ...
      13.4500 32.9500 30.2095 16.1905 ...
      48.0000 48.0000 48.0000 48.0000 ...
      200.0000 26.2095 20.1905 ...
      20.0000 0 4.0000 0];

% --- 2. 设定仿真环境参数 ---
f0 = 50:10:2000;      
nlayer = 2; 
ncell = 4; 
nw = 2; 
delta = 1.0;          
mnum = 500; 
Norder = 9;           

% --- 3. 定义优化边界 (Bounds) ---
lb = zeros(1, 31); ub = zeros(1, 31);
% 管厚度 t (1-8)
lb(1:8) = 1; ub(1:8) = 20;
% 孔直径 d (9-16) - 物理约束在 nonlcon 中严格处理，这里给宽泛范围
lb(9:16) = 1.0; ub(9:16) = 30;
% 单元高度 h (17-20)
lb(17:20) = 5; ub(17:20) = 48;
% 上层腔长 Lu (21-24) - 强制锁定为 48
lb(21:24) = 48; ub(21:24) = 48;
% 总厚度 Ltot (25)
lb(25) = 50; ub(25) = 200;
% 主腔宽度 w (26-27)
lb(26:27) = 5; ub(26:27) = 48;
% 层间板厚 tu (28-31)
lb(28:31) = 0; ub(28:31) = 20;

% --- 4. 定义线性等式约束 (Aeq * x = beq) ---
% 约束1: h1 + h2 = 46.4
% 约束2: h3 + h4 = 46.4
% 约束3: w1 + w2 = 46.4
Aeq = zeros(3, 31);
beq = zeros(3, 1);

Aeq(1, 17) = 1; Aeq(1, 18) = 1; beq(1) = 46.4; 
Aeq(2, 19) = 1; Aeq(2, 20) = 1; beq(2) = 46.4;
Aeq(3, 26) = 1; Aeq(3, 27) = 1; beq(3) = 46.4;

% --- 5. 配置并运行 fmincon 求解器 ---
fprintf('开始反演优化，强制逼近平均吸声系数 0.9，并修复空间越界问题...\n');
options = optimoptions('fmincon', ...
    'Display', 'iter', ...       % 打印迭代过程
    'Algorithm', 'sqp', ...      % 使用序贯二次规划算法，适合非线性约束
    'MaxFunctionEvaluations', 2000, ...
    'StepTolerance', 1e-4);

% 定义匿名目标函数：让平均吸声系数尽可能接近 0.9
obj_func = @(x) abs(0.9 - mean(fun_structure_broadband_nlayer_HB_func(f0, x, ncell, nw, delta, nlayer, mnum, Norder)));

% 执行优化
[sl_opt_fixed, fval] = fmincon(obj_func, x0, [], [], Aeq, beq, lb, ub, @my_physical_constraints, options);

% --- 6. 验证优化后的结果并绘图 ---
fprintf('\n优化完成！正在计算修复后结构的吸声系数...\n');
[alpha_fixed, Z_fixed] = fun_structure_broadband_nlayer_HB_func(f0, sl_opt_fixed, ncell, nw, delta, nlayer, mnum, Norder);

figure;
plot(f0, alpha_fixed, 'r-', 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
title('Constraint-Optimized Structure - Absorption Performance');
ylim([0 1.1]);

% 打印结果
fprintf('=== 优化结果 ===\n');
fprintf('目标函数残差 (越小越接近 0.9): %.4f\n', fval);
fprintf('修复后频段内的平均吸声系数为: %.4f\n', mean(alpha_fixed));
disp('修复后合法的 31 维参数数组为:');
disp(sl_opt_fixed);


% =========================================================================
% 附加函数：定义复杂的非线性物理干涉约束 (防止穿模的核心)
% =========================================================================
function [c, ceq] = my_physical_constraints(x)
    % 非线性不等式约束 c(x) <= 0
    % 非线性等式约束 ceq(x) = 0
    ceq = []; 
    c = zeros(8, 1);
    
    % 提取宽度和高度以便于阅读
    w1 = x(26); w2 = x(27);
    h1 = x(17); h2 = x(18); h3 = x(19); h4 = x(20);
    
    % 约束：每个孔径 d 必须满足 d <= min(w, h) - 4
    % 即：d - min(w, h) + 4 <= 0
    % 第一层孔径 (1-4)
    c(1) = x(9)  - min(w1, h1) + 4;
    c(2) = x(10) - min(w1, h2) + 4;
    c(3) = x(11) - min(w2, h3) + 4;
    c(4) = x(12) - min(w2, h4) + 4;
    
    % 第二层孔径 (5-8)
    c(5) = x(13) - min(w1, h1) + 4;
    c(6) = x(14) - min(w1, h2) + 4;
    c(7) = x(15) - min(w2, h3) + 4;
    c(8) = x(16) - min(w2, h4) + 4;
end