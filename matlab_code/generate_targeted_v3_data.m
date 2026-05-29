% generate_targeted_v3_data.m (V3 平衡数据集生成脚本)

clc, clear

% --- 核心配置 ---
N = 600000; % V3 目标样本总数 (可以设为 60 万或 70 万)

nlayer = 2;
ncell = 4;
nw = 2;

% --- 其他参数 (保持不变) ---
delta = 0.7; 
H = 49.4;
W = 49.4;
Ltotmax = 200;
lbw=4; ubw=48; lbd=1; ubd=30;
lbt=1; ubt=20; lbtu=0; ubtu=20;
lbLtot=50; ublim=[ubw, 48, ubt, ubtu, ubd, 200, 0.9, 0.9];
Ratio_t_ub=0.9; Ratio_L_ub=0.9; Ratio_t_lb=0.1; Ratio_L_lb=0.1;
lblim=[lbw, 48, lbt, lbtu, lbd, lbLtot, Ratio_t_lb, Ratio_L_lb];

% --- 【核心修改 1/2】调用全新的 V3 参数生成函数 ---
disp('--- 正在生成 V3 平衡参数 (目标：均衡 d 参数) ---');
[slN] = Random_sl_mutilcavs_nlayer_func_31params_v3(N, nlayer, ncell, nw, delta, H, W, lblim, ublim);
disp(['生成的新 slN (V3) 数据维度: ', num2str(size(slN))]);

% --- 频率和仿真部分 (保持不变) ---
fmin=60; fstep=1; fmax=3500;
f0=fmin:fstep:fmax;
fL=length(f0);
mnum=500; Norder=9;
alphaM=zeros(fL,N);
ZM=zeros(fL,N);

disp('正在进行大规模并行仿真 (V3)...');
tic
parfor n=1:N
    sl=slN(n,:);
    [alpha,Z]=fun_structure_broadband_nlayer_HB_func(f0,sl,ncell,nw,delta,nlayer,mnum,Norder);
    alphaM(:,n)=alpha;
    ZM(:,n)=Z;
end
toc

RZ=real(ZM);
XZ=imag(ZM);

% --- 【核心修改 2/2】使用新的 V3 文件名保存 ---
disp('正在保存 V3 平衡增强数据集...');
save("2layer_2_2_31params_1Hz_enhanced_v3_slN.txt","slN","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v3_alphaM.txt","alphaM","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v3_RZM.txt","RZ","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v3_XZM.txt","XZ","-ascii")
disp('V3 平衡增强数据集保存完成！');