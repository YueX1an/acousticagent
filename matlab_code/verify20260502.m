% --- 1. 定义优化的 31 维参数 ---
sl_opt = [8.0000, 2.6875, 1.3750, 2.6875, 1.0125, 13.6531, 1.0125, 9.1449, ... % t
         10.9750, 10.2250, 10.2563, 11.2023, 21.2588, 8.8620, 13.4662, 5.4523, ... % d
         25.2588, 21.4412, 17.4662, 29.2338, ... % h
         49.9580, 20.7096, 50.0000, 31.1604, ... % Lu
         57.0000, ...                         % Ltot
         26.3005, 20.3995, ...                 % w
         0.0112, 14.8874, 0.1250, 15.2180
];                                     

% --- 2. 设定仿真环境参数 ---
f0 = 50:10:2000;      % 频率范围 50-2000Hz[cite: 2]
nlayer = 2; 
ncell = 4; 
nw = 2; 
delta = 1.0;          %- 壁厚[cite: 2]
mnum = 500; 
Norder = 9;           % 仿真阶数[cite: 2]

% --- 3. 调用核心仿真函数 ---
fprintf('正在计算优化结构的吸声系数...\n');
[alpha, Z] = fun_structure_broadband_nlayer_HB_func(f0, sl_opt, ncell, nw, delta, nlayer, mnum, Norder);%[cite: 2]

% --- 4. 绘图验证 ---
figure;
plot(f0, alpha, 'b-', 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
title('Optimized Structure (49.4mm) - Absorption Performance');
ylim([0 1.1]);

% 打印平均值
fprintf('在该频段内的平均吸声系数为: %.4f\n', mean(alpha));