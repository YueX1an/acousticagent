% al_step2_matlab_AUTO.m
% 描述: 自动化流程 Step 2 - 读取中转站数据 -> 物理仿真 -> 写回中转站
% 适配目录: 31params (拥有物理函数) <-> AL_Data_Exchange (拥有数据)

clc; clear;

% ==================== 【路径配置】 ====================
% 1. 数据中转站 (绝对路径，必须与 Python 总指挥一致)
exchange_dir = 'C:\Users\leozhang\Desktop\AL_Data_Exchange';

% 2. 输入/输出文件路径
x_file = fullfile(exchange_dir, 'X_current_round.txt');
y_file = fullfile(exchange_dir, 'Y_current_round.txt');
% ====================================================

% --- 1. 检查输入 ---
if exist(x_file, 'file') ~= 2
    error(['错误: 中转站里没找到 X 文件！路径: ' x_file]);
end

% 检查物理函数是否存在 (双重保险)
if exist('fun_structure_broadband_nlayer_HB_func', 'file') ~= 2
    error('错误: 当前目录下找不到物理仿真函数 fun_...m！');
end

% --- 2. 加载数据 ---
X_data = load(x_file);
[N_samples, ~] = size(X_data);
fprintf('>>> MATLAB: 收到 %d 个样本，开始计算...\n', N_samples);

% 仿真参数
nlayer=2; ncell=4; nw=2; delta=0.7; mnum=500; Norder=9;
f0 = linspace(0, 2000, 100); 

Y_true = zeros(N_samples, 100);

% --- 3. 批量计算 (含 0Hz 修复) ---
% 为了速度，我们关闭非必要的警告
warning('off', 'MATLAB:singularMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');

tic;
parfor i = 1:N_samples  % 如果您有并行工具箱，可用 parfor；否则改回 for
    try
        % 运行物理仿真
        [alpha, ~] = fun_structure_broadband_nlayer_HB_func(f0, X_data(i,:), ncell, nw, delta, nlayer, mnum, Norder);
        
        % 【关键修复】处理 0Hz 处的 NaN
        if any(isnan(alpha))
            alpha(isnan(alpha)) = 0;
        end
        
        % 再次检查是否全为 Inf 或 NaN (真正的计算失败)
        if any(isinf(alpha))
            Y_true(i, :) = zeros(1, 100);
        else
            Y_true(i, :) = alpha;
        end
    catch
        Y_true(i, :) = zeros(1, 100);
    end
end
toc;

warning('on', 'all');

% --- 4. 保存结果并退出 ---
save(y_file, 'Y_true', '-ascii');
fprintf('>>> MATLAB: 计算完成，结果已保存至: %s\n', y_file);

% 注意：不要写 exit，Python 的 -batch 参数会在脚本跑完后自动处理退出