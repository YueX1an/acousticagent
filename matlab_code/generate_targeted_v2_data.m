% generate_targeted_v2_data.m (第二轮靶向生成主脚本)

clc, clear

% --- 核心配置 ---
N = 100000; % 生成10万条新的“第二轮靶向”数据
HARD_CASE_RATIO = 0.7; % 【可调整】第二轮可以适当提高困难样本比例，例如60%

nlayer = 2; ncell = 4; nw = 2;
delta = 0.7; H = 49.4; W = 49.4;
lbw=4; ubw=48; lbd=1; ubd=30; lbt=1; ubt=20; lbtu=0; ubtu=20; lbLtot=50; ubLtot=200;
ublim=[ubw, 48, ubt, ubtu, ubd, ubLtot, 0.9, 0.9];
lblim=[lbw, 48, lbt, lbtu, lbd, lbLtot, 0.1, 0.1];

% --- 【核心修改 1/2】调用第二轮靶向函数 ---
disp('正在生成第二轮靶向参数...');
[slN] = Random_sl_mutilcavs_nlayer_func_31params_TARGETED_v2(N, nlayer, ncell, nw, delta, H, W, lblim, ublim, HARD_CASE_RATIO);
disp(['生成的新 slN 数据维度: ', num2str(size(slN))]);

% --- 仿真与保存部分 (保持不变，仅修改文件名) ---
fmin=60; fstep=1; fmax=3500; f0=fmin:fstep:fmax; fL=length(f0);
mnum=500; Norder=9;
alphaM=zeros(fL,N); ZM=zeros(fL,N);

disp('正在进行大规模并行仿真...');
tic
parfor n=1:N
    sl=slN(n,:);
    [alpha,Z]=fun_structure_broadband_nlayer_HB_func(f0,sl,ncell,nw,delta,nlayer,mnum,Norder);
    alphaM(:,n)=alpha;
    ZM(:,n)=Z;
end
toc

RZ=real(ZM); XZ=imag(ZM);

% --- 【核心修改 2/2】使用新的文件名保存 (v2) ---
disp('正在保存第二轮靶向增强数据集 (v2)...');
save("2layer_2_2_31params_1Hz_enhanced_v2_slN.txt","slN","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v2_alphaM.txt","alphaM","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v2_RZM.txt","RZ","-ascii")
save("2layer_2_2_31params_1Hz_enhanced_v2_XZM.txt","XZ","-ascii")
disp('第二轮靶向增强数据集保存完成！');