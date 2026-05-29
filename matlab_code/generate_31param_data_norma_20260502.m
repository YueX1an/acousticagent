% generate_31param_data_normal.m
% 适配 Ltot=57, 外部尺寸 49.7x49.7x60, 内部参数放开的新策略
clc; clear; close all;

% --- 核心配置 ---
N_total = 500000; % 总样本数 
nlayer = 2; ncell = 4; nw = 2;
delta = 1.0; 
H = 49.7;     % 外部总高 (从 49.4 更新为 60)
W = 49.7;     % 外部总宽 (从 49.4 更新为 49.7)

% 提取自界面的优化限制边界
lbw = 15;   ubw = 40;    % 单元宽度 (w)
lbLu = 15;  ubLu = 40;   % 单元长度/上层腔长 (Lu)
lbd = 1.0;  ubd = 30;    % 内插管径 (d)
lbt = 1;    ubt = 200;   % 内插管长 (t) - 允许长管折叠
lbtu = 0;   ubtu = 200;  % 上向接管长 (tu)
lbLtot= 57; ubLtot= 57;  % 总高度 (Ltot) - 严格锁死在 57

% 组装边界数组供生成器使用
% 数组索引含义: [w, Lu, t, tu, d, Ltot, Ratio_t, Ratio_L]
ublim = [ubw, ubLu, ubt, ubtu, ubd, ubLtot, 0.9, 0.9];
lblim = [lbw, lbLu, lbt, lbtu, lbd, lbLtot, 0.1, 0.1];

% --- 调用新的生成器 ---
disp(['--- 正在生成受限空间 (Ltot=57) 数据集 (N=', num2str(N_total), ') ---']);
OutParams = Random_sl_mutilcavs_nlayer_func_31params_fixed_20260502(N_total, nlayer, ncell, nw, delta, H, W, lblim, ublim);

% 将独立参数结构体横向拼接回 Nx31 的矩阵
disp('正在将独立参数转换为 slN (Nx31) 矩阵格式以供仿真调用...');
slN = [OutParams.td, OutParams.d, OutParams.h, OutParams.Lu, OutParams.Ltot, OutParams.w, OutParams.tu];
disp(['生成的新 slN 数据维度: ', num2str(size(slN))]);

% --- 频率和仿真部分 ---
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

% --- 保存部分 ---
disp('正在保存正常数据集...');
save("2layer_2_2_31params_10Hz_slN.txt","slN","-ascii")
save("2layer_2_2_31params_10Hz_alphaM.txt","alphaM","-ascii")
save("2layer_2_2_31params_10Hz_RZM.txt","RZ","-ascii")
save("2layer_2_2_31params_10Hz_XZM.txt","XZ","-ascii")

disp('正常数据集保存完成！');