% generate_31param_data_normal.m
% 生成 600,000 个正常样本，使用最新修改的 fixed 生成器
clc; clear; close all;

% --- 核心配置 ---
N_total = 500000; % 总样本数 600,000
nlayer = 2; ncell = 4; nw = 2;
delta = 1.0; H = 49.4; W = 49.4;
Ltotmax = 200;

lbw=5; ubw=48; lbd=1.0; ubd=30;
lbt=1; ubt=20; lbtu=0; ubtu=20;
lbLtot=50; 
ublim=[ubw, 48, ubt, ubtu, ubd, 200, 0.9, 0.9];
Ratio_t_ub=0.9; Ratio_L_ub=0.9; Ratio_t_lb=0.1; Ratio_L_lb=0.1;
lblim=[lbw, 48, lbt, lbtu, lbd, lbLtot, Ratio_t_lb, Ratio_L_lb];

% --- 调用新的生成器 (Normal Only) ---
disp(['--- 正在生成常规数据集 (N=', num2str(N_total), ') ---']);

% 调用 fixed 函数（返回结构体）
OutParams = Random_sl_mutilcavs_nlayer_func_31params_fixed(N_total, nlayer, ncell, nw, delta, H, W, lblim, ublim);

% 为兼容后续仿真函数，在此处将独立参数结构体横向拼接回 Nx31 的矩阵
disp('正在将独立参数转换为 slN (Nx31) 矩阵格式以供仿真调用...');
slN = [OutParams.td, OutParams.d, OutParams.h, OutParams.Lu, OutParams.Ltot, OutParams.w, OutParams.tu];

disp(['生成的新 slN 数据维度: ', num2str(size(slN))]);

% --- 频率和仿真部分 (保持不变) ---
fmin=50; fstep=10; fmax=2000;
f0=fmin:fstep:fmax;
fL=length(f0);
mnum=500; Norder=9;

alphaM=zeros(fL,N_total);
ZM=zeros(fL,N_total);

disp('正在进行大规模并行仿真...');
tic
parfor n=1:N_total 
    sl=slN(n,:);
    [alpha,Z]=fun_structure_broadband_nlayer_HB_func(f0,sl,ncell,nw,delta,nlayer,mnum,Norder);
    alphaM(:,n)=alpha;
    ZM(:,n)=Z;
end
toc

RZ=real(ZM);
XZ=imag(ZM);

% --- 保存部分 (文件名去掉了 mixed_v3) ---
disp('正在保存正常数据集...');
save("2layer_2_2_31params_10Hz_slN.txt","slN","-ascii")
save("2layer_2_2_31params_10Hz_alphaM.txt","alphaM","-ascii")
save("2layer_2_2_31params_10Hz_RZM.txt","RZ","-ascii")
save("2layer_2_2_31params_10Hz_XZM.txt","XZ","-ascii")

disp('正常数据集保存完成！');