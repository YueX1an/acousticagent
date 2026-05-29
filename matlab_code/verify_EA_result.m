% verify_EA_result.m
%
% 描述:
%   用于物理验证 Python 逆向设计 (EA) 找到的
%   "真实多窄带 (Index 8)" 样本。
%   此脚本将对比 *真实* 参数 和 *找到* 参数 
%   在 MATLAB 中的物理仿真结果。

clc; clear; close all;

disp('--- 开始物理验证：EA 非闭环测试 (Index 8) ---');

% --- 1. 设置仿真参数 (保持不变) ---
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; mnum = 500; Norder = 9;

% --- 2. 设置频率范围 (保持不变) ---
fmin=60; fstep=1; fmax=3500;
f0=fmin:fstep:fmax;
fL = length(f0);
disp(['仿真频率范围: ' num2str(fmin) 'Hz to ' num2str(fmax) 'Hz']);

% --- 3. 【核心】定义“真实的”和“找到的”两组参数 ---

% 真实的结构参数 X_true (来自日志)
sl_true = [
     3.4734628,   3.7483013,   5.11061,     2.8097072,   5.609743,   13.067544, ...
     6.548911,   17.362698,   2.2972713,   2.3962164,  14.543285,    3.7711577, ...
     2.5304456,   1.754544,    7.7494583,   7.7071958,  22.578993,   41.120995, ...
    31.879028,   18.563944,   27.394028,   26.718523,   20.773438,   25.780506, ...
    74.93076,     9.026006,   38.273994,   10.904018,   12.344894,    6.0080705, ...
     0.79266554
];

% 优化后的结构参数 X_optimized (来自日志)
sl_optimized = [
    19.66189181, 17.43013466,  1.1571267,   3.93489046,  3.36388689,  8.223369, ...
    19.98957307, 10.20063055,  1.37216076,  1.9689545,  16.02028146,  3.66486668, ...
    26.9493232,  18.92249282,  8.62220374, 14.64420051, 29.78182746, 26.47262537, ...
    43.24157232, 33.2810028,  22.1457205,  21.64908049, 26.26628471, 22.83898087, ...
    69.84526062, 21.78340958, 34.42892705,  7.72685395,  7.30629703, 11.38807779, ...
    11.22623788
];
disp('已加载 Python (EA) 找到的 X_optimized 和 X_true 参数。');

% --- 4. 运行两次物理仿真 ---
disp('正在运行 MATLAB 物理仿真 (fun_structure_broadband_nlayer_HB_func)...');
tic;
% 运行 "真实" 曲线 (A)
[alpha_true, Z_true] = fun_structure_broadband_nlayer_HB_func(f0, sl_true, ncell, nw, delta, nlayer, mnum, Norder);
% 运行 "找到" 曲线 (B)
[alpha_optimized, Z_optimized] = fun_structure_broadband_nlayer_HB_func(f0, sl_optimized, ncell, nw, delta, nlayer, mnum, Norder);
toc;
disp('两次物理仿真完成。');

% --- 5. 绘图对比 ---
disp('正在绘制对比图...');
figure;
hold on;
% 绘制目标
plot(f0, alpha_true, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Target (Physical Alpha from X_true)');
% 绘制 EA 找到的结果
plot(f0, alpha_optimized, 'r--', 'LineWidth', 2, 'DisplayName', 'Physical Result (from X_optimized)');

title('Verification: EA Non-Closed-Loop Test (Index 8)', 'FontSize', 14);
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient (Alpha)');
ylim([-0.05, 1.1]);
grid on;
legend('show', 'Location', 'northwest');
hold off;

disp('--- 验证完成 ---');