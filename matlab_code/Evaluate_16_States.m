% =========================================================================
% Evaluate_16_States.m
% 评估 16 个精英结构的真实吸声光谱
% =========================================================================
clc; clear;
disp('正在读取 16 态字典并提取物理光谱...');

% 1. 加载参数字典
if ~isfile('elite_16_dictionary.txt')
    error('未找到 elite_16_dictionary.txt，请确保在同一目录下。');
end
elite_16 = load('elite_16_dictionary.txt');

% 2. 仿真环境
f0 = 50:10:2000;
freq_targets = [400, 900, 1400, 1900];
[~, idx] = min(abs(f0' - freq_targets), [], 1); % 找准 4 个频点的索引

nlayer = 2; ncell = 4; nw = 2; delta = 1.0; mnum = 500; Norder = 9;
alpha_16 = zeros(16, 4);

% 3. 并行仿真 (飞速跑完 16 个)
parfor i = 1:16
    alpha = fun_structure_broadband_nlayer_HB_func(f0, elite_16(i,:), ncell, nw, delta, nlayer, mnum, Norder);
    alpha_16(i, :) = alpha(idx);
end

% 4. 导出给 Python 用
save('elite_16_spectra.txt', 'alpha_16', '-ascii');
disp('✅ 16 态物理光谱已保存为 elite_16_spectra.txt！准备交接给 Python。');