% al_step2_matlab_FIXED.m
% 描述: 主动学习 Step 2 (修复版) - 修复 0Hz 导致的 NaN 问题。

clc; clear; close all;

% --- 配置 ---
data_dir = 'active_learning_round1';
x_file = fullfile(data_dir, 'AL_X_candidates.txt');

if exist(x_file, 'file') ~= 2
    error('找不到 X 参数文件，请先运行 Python Step 1 脚本！');
end

% 检查核心函数
if exist('fun_structure_broadband_nlayer_HB_func', 'file') ~= 2
    error('请确保 fun_structure_broadband_nlayer_HB_func.m 在当前文件夹！');
end

X_data = load(x_file);
[N_samples, ~] = size(X_data);
disp(['--- AL Step 2: 正在计算 ', num2str(N_samples), ' 个样本 ---']);

% 仿真参数
nlayer=2; ncell=4; nw=2; delta=0.7; mnum=500; Norder=9;
f0 = linspace(0, 2000, 100); 

Y_true = zeros(N_samples, 100);

% 进度条
fprintf('进度: ');

tic;
for i = 1:N_samples
    if mod(i, 100) == 0, fprintf('%d..', i); end
    
    try
        % 运行物理仿真
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(f0, X_data(i,:), ncell, nw, delta, nlayer, mnum, Norder);
        
        % 【核心修复】将 NaN (通常是 0Hz 处) 替换为 0
        alpha(isnan(alpha)) = 0;
        
        % 再次检查：只有当非 0Hz 处出现 Inf 时才视为失败
        if any(isinf(alpha))
            Y_true(i, :) = zeros(1, 100); % 真正失败
        else
            Y_true(i, :) = alpha; % 保存修复后的数据
        end
        
    catch
        Y_true(i, :) = zeros(1, 100);
    end
end
toc;
fprintf('\n');

% 保存真值
output_file = fullfile(data_dir, 'AL_Y_true.txt');
save(output_file, 'Y_true', '-ascii');

% 验证一下文件是否为空
data_check = load(output_file);
valid_count = sum(max(data_check, [], 2) > 0.01);
disp(['✅ 计算完成。有效样本数: ' num2str(valid_count) ' / ' num2str(N_samples)]);