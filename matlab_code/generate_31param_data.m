% generate_31param_data.m (已更新为 V3 混合生成方案)
% 【已修正】生成 600,000 个样本用于最终 V3 数据集
% 【已修正】绘图部分已注释掉
% 【已修正】保存文件名已更新

clc; clear; close all;

% --- 核心配置 ---
N_total = 600000; % 【核心修改】总样本数改为 600,000
N_normal = N_total / 2; % 300,000 个常规样本
N_elite = N_total / 2;  % 300,000 个精英偏向样本

nlayer = 2; ncell = 4; nw = 2;
delta = 0.7; H = 49.4; W = 49.4;
Ltotmax = 200;
lbw=4; ubw=48; lbd=1; ubd=30;
lbt=1; ubt=20; lbtu=0; ubtu=20;
lbLtot=50; ublim=[ubw, 48, ubt, ubtu, ubd, 200, 0.9, 0.9];
Ratio_t_ub=0.9; Ratio_L_ub=0.9; Ratio_t_lb=0.1; Ratio_L_lb=0.1;
lblim=[lbw, 48, lbt, lbtu, lbd, lbLtot, Ratio_t_lb, Ratio_L_lb];

% --- 【核心修改 1/4】调用两种生成器 ---
disp(['--- 正在生成 V3 混合数据集 (N=', num2str(N_total), ') ---']);

% 1. 生成 50% V1 常规样本
disp(['1. 正在生成 ', num2str(N_normal), ' 个 V1 "常规" 样本...']);
[slN_normal] = Random_sl_mutilcavs_nlayer_func_31params(N_normal, nlayer, ncell, nw, delta, H, W, lblim, ublim);

% 2. 生成 50% V3 精英偏向样本
disp(['2. 正在生成 ', num2str(N_elite), ' 个 V3 "精英偏向" 样本...']);
[slN_elite] = Random_sl_mutilcavs_nlayer_func_31params_v3_elite(N_elite, nlayer, ncell, nw, delta, H, W, lblim, ublim);

% 3. 合并
disp('3. 正在合并数据集 (不打乱)...');
slN = [slN_normal; slN_elite]; % (保持不打乱，以便后续绘图)
clear slN_normal slN_elite; % 释放内存
% 【注意】我们将在 Python 中进行最终的打乱，而不是在 MATLAB 中
% slN = slN(randperm(N_total), :); % (移除打乱以节省内存和时间)
disp(['生成的新 slN (V3 混合) 数据维度: ', num2str(size(slN))]);

% --- 频率和仿真部分 (保持不变) ---
fmin=60; fstep=1; fmax=3500;
f0=fmin:fstep:fmax;
fL=length(f0);
mnum=500; Norder=9;
alphaM=zeros(fL,N_total);
ZM=zeros(fL,N_total);
disp('正在进行大规模并行仿真 (V3 混合)...');
tic
parfor n=1:N_total %
    sl=slN(n,:);
    [alpha,Z]=fun_structure_broadband_nlayer_HB_func(f0,sl,ncell,nw,delta,nlayer,mnum,Norder);
    alphaM(:,n)=alpha;
    ZM(:,n)=Z;
end
toc
RZ=real(ZM);
XZ=imag(ZM);

% --- 【核心修改 4/4】绘图验证 (已注释掉) ---
% disp('正在绘制 V3 生成结果验证图 (平均曲线)...');
% ...
% --- 绘图结束 ---

% --- 【核心修改】使用最终 V3 文件名保存 ---
disp('正在保存 V3 混合数据集...');
save("2layer_2_2_31params_1Hz_mixed_v3_slN.txt","slN","-ascii")
save("2layer_2_2_31params_1Hz_mixed_v3_alphaM.txt","alphaM","-ascii")
save("2layer_2_2_31params_1Hz_mixed_v3_RZM.txt","RZ","-ascii")
save("2layer_2_2_31params_1Hz_mixed_v3_XZM.txt","XZ","-ascii")
disp('V3 混合数据集保存完成！');