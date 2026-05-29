%%%% 王能银 2021/12/25 结构依旧利用阻抗计算但是更加简写加入对不上就管内展开重新计算阻抗 %%%%
%%%% 王能银 2021/12/29 对的还可以准备优化 将函数写的尽量快速 只有一个求解是否可以将频率写成频带一起求解？ %%%%
%%%% 王能银 2022/1/2 三层优化 sl取一位小数 %%%%
%%%% 王能银 2022/1/5 三层优化 画图 主程序 %%%%
%%%% 王能银 2022/11/29 高频考虑高阶入射 %%%%
%%%% 王能银 2022/12/9 硬边界条件 %%%%
%%%% 王能银 2025/5/4 函数 %%%%

function [alpha_a,Z]=fun_structure_broadband_nlayer_HB_func(f0,sl,ncell,nw,delta,nlayer,mnum,Norder)

f0 = double(f0);          % 防御: 确保double (Python .mat 可能为int64)
Nmax=Norder;
% nmax=ceil(Nmax/2);%外部区域y方向阶数%%合适的阶数，9阶一下不太好！
Mmax=Norder;
% mmax=ceil(Mmax/2);%外部区域x方向阶数
% 
% c0=931.4;%空气声速 1000℃
% rho0=0.277;%空气密度  1000℃
% c0=511.4;%空气声速 300℃
% rho0=0.616;%空气密度  300℃
% c0=391.4;%空气声速 100℃
% rho0=0.946;%空气密度  100℃
c0=343.2;%空气声速 20℃
rho0=1.2043;%空气密度  20℃
% Tssd=1000;
% c0=331.4+0.6*Tssd
% Pair=101325;%标准大气压
% Rair=287;%空气气体常数
% rho0=Pair/(Rair*(Tssd+273.15))

Z0=rho0*c0;
% rho1=rho0;%外部区域I密度
omega=2*pi.*f0;
k0=2*pi.*f0/ c0;%波矢
% thetai=0/180*pi;%入射角度 与轴夹角且在xoz平面内
%%

%%% 注意：sl参数被四舍五入到小数点后一位了，结构参数的实际值是被离散化处理的
%%% 可能对后续神经网络学习产生影响，因为神经网络尝试学习一个一个连续的映射，而不是面对一个内在离散的物理系统
sl = double(sl);           % 防御: 确保double精度 (Python .mat 可能为single)
sl = round(sl, 1);         % 舍入到小数点后一位 (等效 roundn(sl,-1), 无需Mapping Toolbox)
sl = sl * 1e-3;            % mm → m
% [c,d,xx,yy,zs,a]=zs_square_grating_nlayer(sl,ncell,nw,nlayer,delta,f0);
% [c,d,xx,yy,zs,a]=zs_square_grating_nlayer_mnum(sl,ncell,nw,nlayer,delta,f0,mnum);%350 0.898
% [c,d,xx,yy,zs,a]=zs_square_grating_nlayer_mnum_tend(sl,ncell,nw,nlayer,delta,f0,mnum);
% [c,d,xx,yy,zs,a]=zs_square_grating_multilayer_mnum(sl,ncell,nw,nlayer,delta,f0,mnum);
[c,d,xx,yy,zs,a]=zs_square_grating_multilayer_mnum_amend(sl,ncell,nw,nlayer,delta,f0,mnum);
Ss=c.*d;
zs=-zs;
% zs=zs*10000000;
% singlei=1;
% zs(:,singlei)=zs(:,singlei)/10000000;

% a=0.05;
b=a;
% b=2*a;
S=a*b;
MM=1:Mmax;%-M:M
NN=1:Nmax;%-M:M
kx=(MM-1)*pi/a;
ky=(NN-1)*pi/b;

delta_m1=1/2*ones(Mmax,1);
delta_m1(1)=1;
delta_n1=1/2*ones(Nmax,1);
delta_n1(1)=1;
delta_m1n1=kron(delta_n1,delta_m1);



fL=length(f0);
alpha_a=zeros(fL,1);
r0=zeros(1,fL);

% X_fun=@(x) cos(kx.*x);
% XX_fun=@(x) X_fun(x).*(X_fun(x)).';
% Y_fun=@(y) cos(ky.*y);
% YY_fun=@(y) Y_fun(y).*(Y_fun(y)).';

MMpia=((MM-1)+(MM-1).')*pi/a;
M_Mpia=(MM-MM.')*pi/a;
M2_pia=(MM+MM-2)*pi/a;
NNpib=((NN-1)+(NN-1).')*pi/b;
N_Npib=(NN-NN.')*pi/b;
N2_pib=(NN+NN-2)*pi/b;

for fn=1:fL
    %% The wave vector of free space 
    sign=(-1).^(((abs(kx)).^2+(abs(ky.')).^2<=(abs(k0(fn))).^2)+1);
    kz=sqrt(k0(fn).^2-kx.^2-ky.'.^2).*sign;
    thetar=acos(kz./k0(fn)).*(1+sign)/2+pi/2.*(1-sign)/2;
    %%  Input matrix
    Kz=reshape(kz.',[],1);
    Kz_deltamn=Kz.*delta_m1n1;
    N1=diag(Kz_deltamn);

%     delta_kx=kx-kx.';
%     delta_ky=ky-ky.';
    N2=zeros(Nmax*Mmax,Nmax*Mmax,ncell);%square
    for i=1:ncell%速度提升！！
        N2x=sin(MMpia*c(i)/2).*cos(MMpia*xx(i))./(MMpia*c(i))...
            +sin(M_Mpia*c(i)/2).*cos(M_Mpia*xx(i))./(M_Mpia*c(i));
        N2x(isnan(N2x))=sin(M2_pia*c(i)/2).*cos(M2_pia*xx(i))./(M2_pia*c(i))...
            +1/2;
        N2x(1,1)=1;
        N2y=sin(NNpib*d(i)/2).*cos(NNpib*yy(i))./(NNpib*d(i))...
            +sin(N_Npib*d(i)/2).*cos(N_Npib*yy(i))./(N_Npib*d(i));
        N2y(isnan(N2y))=sin(N2_pib*d(i)/2).*cos(N2_pib*yy(i))./(N2_pib*d(i))...
            +1/2;
        N2y(1,1)=1;

%         N2x=1/c(i)*integral(XX_fun,xx(i)-c(i)/2,xx(i)+c(i)/2,'ArrayValued',true,'abstol',1e-12);
%         N2y=1/d(i)*integral(YY_fun,yy(i)-d(i)/2,yy(i)+d(i)/2,'ArrayValued',true,'abstol',1e-12);
        % --- 【补丁 1：物理边界保护，防止绝对零阻抗导致除零溢出】 ---
        zs_safe = zs(fn,i);
        if abs(zs_safe) < 1e-10
            zs_safe = 1e-10 * (1 - 1i); % 引入极微小的数值阻尼，柔化奇异峰
        end
        N2(:,:,i)=rho0*omega(fn)/zs_safe*Ss(i)/S.*kron(N2y,N2x);
    end
    N2II=sum(N2,3);
    %%
    A=N1-N2II;
    % AminusM=zeros(Nmax,Mmax);
    % AminusM(1,3)=1;
    % Aminus=reshape(AminusM,Nmax*Mmax,1);
    Aminus=zeros(Nmax*Mmax,1);
    Aminus(1,1)=1;
    % Aminus=reshape((sign+1).'/2,[],1);% all high order incidence with A=1;
    % Aminus(0+1,1)=1;
    % AminusM=reshape(Aminus,Nmax,Mmax);
    bb=(N2II+N1)*Aminus;
    % --- 【补丁 2：鲁棒线性求解，防止高阶虚模态导致矩阵奇异】 ---
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:illConditionedMatrix');
    
    % 如果矩阵条件数极差 (比如包含高阶共振衰减截断误差)，启用防崩解法
    if rcond(A) < 1e-15 || isnan(rcond(A))
        % lsqminnorm 是专治病态矩阵的最小范数解法，不会产生 NaN
        X = lsqminnorm(A, bb); 
    else
        X = A \ bb;
    end
    Aplus=X(1:Nmax*Mmax);
    % AplusM=reshape(Aplus,Nmax,Mmax).*(sign+1).'/2;
    % rabs=reshape(abs(Aplus),Nmax,Mmax).*(sign+1)/2;
    % rM=reshape(Aplus,Nmax,Mmax).*(sign+1)/2;    
    % rpha=angle(AplusM)/pi*180;
    R=Aplus(1:Mmax);
    r0(fn)=R(1)/Aminus(1,1);
%     r0(fn)=R(mmax);
    
%     Rrr=zeros(Nmax,Mmax);
%     for n=1:Nmax
%         for m=1:Mmax
%             if real(kz(n,m))>0 && imag(kz(n,m))==0
%                 Rrr(n,m)=abs(Aplus((n-1)*Mmax+m))^2*cos(thetar(n,m))/cos(thetai);
% %                 thetar(n,m)=thetar(n,m)/pi*180; 
%             else
% %                 thetar(n,m)=90;
%                 Rrr(n,m)=0;
%             end
%         end
%     end
%     Rrrtot=sum(sum(Rrr));
% 
%     I_out_R=Rrrtot/(2*rho0*c0);
%     I_in=1/(2*rho0*c0);
%     R_1=I_out_R/I_in;
%     alpha_a(fn)=1-R_1;
%     W_in=-1/2*real(conj(1./(rho1*omega(fn))).*abs(Aminus.').^2*conj(Kz));
%     W_re=1/2*real(conj(1./(rho1*omega(fn))).*abs(Aplus.').^2*conj(Kz));
    cos_thetar=cos(reshape(thetar.',[],1));
%     W_in=-abs(Aminus.').^2/(2*Z0)*cos_thetar;
%     W_re=abs(Aplus.').^2/(2*Z0)*cos_thetar;
    W_in=-(delta_m1n1.'.*abs(Aminus.').^2)/(2*Z0)*cos_thetar;
    W_re=(delta_m1n1.'.*abs(Aplus.').^2)/(2*Z0)*cos_thetar;
    R_tot=abs(W_re./W_in);
    alpha_a(fn)=1-R_tot;
    % R1=Aplus((nmax-1)*Mmax+mmax);
    % thetar=thetar*180/pi;
end
% alpha=-sum(alpha_a)/fL;
% alpha_min=min(alpha_a);

L0=0;%L0处表面阻抗
R0=r0.*exp(-2i.*k0*L0);
ra=(real((1+R0)./(1-R0))).';
xa=(imag((1+R0)./(1-R0))).';
Z=ra+1i*xa;
end

