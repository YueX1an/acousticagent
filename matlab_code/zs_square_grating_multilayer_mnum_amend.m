%%%% Wangnengyin 2022/3/30 多层合并 精简程序 nlayer>1%%%%
%%%% Wangnengyin 2022/5/19 参数还是用之前的顺序%%%%
%%%% sl=[t,delta_t,d,s,h,L];ncell*(4*nlyaer-1)+nw+1=2*LN+(2*nlayer-1)*Ln+nw+1=4*LN-Ln+nw+1
%%%原先排列 sl=[t,d,h,L,s,delta_t]
% t=[t1,t2,...],ncell*nlayer;
% delta_t=[t12,t23,...],ncell*(nlayer-1);delta_t=tdu;
% d=[d1,d2,...],ncell*nlayer;
% s,nw;
% h,ncell;
% L=[Ltot,L1,L2,...,L(nlayer-1)],ncell*(nlayer-1)+1;%%%%
% L=[L1,L2,...,L(nlayer-1),Ltot],ncell*(nlayer-1)+1;%%%%

function [c,d_end,x,y,zs,a]=zs_square_grating_multilayer_mnum_amend(sl,ncell,nw,nlayer,delta,f,mnum)
%% sl t(nlayer*ncell),d(nlayer*ncell),h(ncell),s(nw),L(nlayer*ncell,Lu,Lm,Ld...) or L((nlayer-1)*ncell+1,Ltot is the total length) ,delta_t((nlayer-1)*ncell),
% sl=[20,30,2,2,4,40,100,4,0];
%alpha_min"=0.892235 1/48 20Hz 打印样品！10Hz平均0.9365
% sl=[0.739772727272727,0.703657262277952,0.722727272727273,0.728636755904299,0.713636363636362,0.815726227795190,0.715608672936260,0.729924242424249,0.704310344827590,0.747283176593522,0.734090909090910,0.702651515151517,0.700000000000002,0.727272727272728,0.740530303030308,0.734090909090920,0.854192789968653,125.336363636364,1.16003134796238,2.84609557109553,220.649242424242,64.9765151515152,0.953787878787880,0.724242424242440,15.9500000000000,171.917803030303,226.229845872518,0.734848484848487,0.700000000000004,0.704545454545477,0.726325757575770,65.0150470219436,2.46875000000001,10.3750000000000,27.8750000000000,115.676240653193,13.2500000000000,30.3125000000000,8.62500000000000,44.1250000000000,19.5000000000000,45.7500000000000,10,28.0500000000000,16.3750000000000,15.8750000000000,116.625000000000,7.37500000000001,6.97500000000000,7.97500000000000,7.97500000000000,8.35000000000000,8.47500000000001,8.47500000000000,8.47500000000000,8.97500000000000,7.97500000000000,7.97500000000000,8.35000000000000,7.78750000000000,7.97500000000000,8.35000000000000,8.35000000000000,8.35000000000000,6.35000000000000,7.85000000000000,7.78750000000000,7.97500000000000,8.35000000000000,3.97500000000000,8.47500000000000,8.72500000000000,7.97500000000000,4.85000000000000,7.60000000000000,7.97500000000000,7.97500000000000,8.35000000000000,8.35000000000000,3.47500000000000,6.03750000000000,6.97500000000000,3.47500000000000,5.22500000000000,8.47500000000000,5.47500000000000,8.47500000000000,3.85000000000000,7.85000000000000,7.97500000000000,2.97500000000000,4.10000000000000,5.47500000000000,3.72500000000000,3.85000000000000,5.47500000000000,10.6625000000000,10.9125000000000,10.9125000000000,13.4125000000000,11.3500000000000,11.3500000000000,11.3500000000000,11.8500000000000,10.8500000000000,10.8500000000000,13.3500000000000,10.8500000000000,10.8500000000000,11.3500000000000,12.3500000000000,11.3500000000000,7.86451802507837,20.5091431556949,8.66242163009404,8.79308311783090,12.4090909090909,93.6643155694880,10.5702716823407,13.2310606060606,44.3857758620690,8.11820794148380,21.0399686520376,8.25662878787879,9.75000000000000,12.5681818181818,16.6950757575758,34.5873824451411,307.135481974922,242.952978056426,305.837578369906,190.799116065677,285.685606060607,213.585684430512,291.179728317660,140.185606060606,216.989224137931,261.131792058516,293.543364681295,306.680871212121,281.500000000000,225.511363636364,158.768939393939,282.662617554859,350,11.2250000000000,12.2250000000000,11.2250000000000,11.2250000000000,1.00227272727273,11.6536572622780,2.04772727272727,0.0536367559043000,3.22613636363636,20.5157262277952,4.54060867293626,1.24242424242425,0.0293103448275900,2.50978317659352,11.5590909090909,5.90265151515151,0.0250000000000000,0.0210227272727210,0.00303030303030560,4.55909090909091,244.854192789968,63.3988636363636,245.410031347962,38.0960955710956,7.89924242424241,83.0390151515151,210.328787878788,65.5992424242424,129.325000000000,10.0428030303030,1.91734587251827,245.609848484849,219.012500000000,163.704545454546,36.5700757575758,56.7650470219440];
% sl=sl*1e-3;

% c0=931.4;%空气声速 1000℃
% rho0=0.277;%空气密度  1000℃
% c0=511.4;%空气声速 300℃
% rho0=0.616;%空气密度  300℃
% c0=391.4;%空气声速 100℃
% rho0=0.946;%空气密度  100℃
% c0=343.2;%空气声速 20℃
% rho0=1.2043;%空气密度  20℃

% Tssd=1000;
% c0=331.4+0.6*Tssd;
% Pair=101325;%标准大气压
% Rair=287;%空气气体常数
% rho0=Pair/(Rair*(Tssd+273.15));

c0=343.2;%20℃
omega=2*pi*f;

k=omega/c0;
rho0=1.2043;%20℃
z00 =rho0*c0;
u=1.81*1e-5;

Ln=ncell;
LN=Ln*nlayer;
% sl=[sl(1:LN),sl(end-LN+Ln+1:end),sl(LN+1:2*LN),sl(3*LN+1+1:3*LN+1+nw),sl(2*LN+1:2*LN+Ln),sl(2*LN+Ln+1:3*LN+1)];
% sl=[sl(1:LN),sl(end-LN+Ln+1:end),sl(LN+1:2*LN),sl(3*LN+1+1:3*LN+1+nw),sl(2*LN+1:2*LN+Ln),sl(2*LN+Ln+1:3*LN+1)];
% sl=[sl(Ln+1:LN),sl(1:Ln),sl(end-LN+Ln+1:end),sl(LN+Ln+1:2*LN),sl(LN+1:LN+Ln),sl(3*LN+1+1:3*LN+1+nw),sl(2*LN+Ln+1:3*LN),sl(2*LN+1:2*LN+Ln),sl(3*LN+1)];

nh=Ln/nw;% width number2
% nvar=(4*nlayer-1)*Ln+nw+1;%LN+LN+Ln+nw+((nlayer-1)*Ln+1)+(nlayer-1)*Ln% variation number
% nvar=4*LN-Ln+nw+1;% variation number

%% 以下所有数据由下往上
% t=sl(1:LN);%length of tube
% tdu=sl(LN+1:2*LN-Ln);%
% delta_t=[zeros(1,Ln),tdu];%delta_t of first(downmost) layer and other layer
% d=sl(2*LN-Ln+1:3*LN-Ln);%diameter of aperture
% d_end=d(end-Ln+1:end);%diameter of aperture up
% s=sl(3*LN-Ln+1:3*LN-Ln+nw);%width of cavity
% h=sl(3*LN-Ln+nw+1:3*LN+nw);%width of cavity
% LL=sl(3*LN+nw+1:4*LN-Ln+nw+1);%
% L_1=reshape(LL(1:end-1),Ln,nlayer-1).';%length of cavity L(1)=Ltot;
% L_tot=LL(end);
% L=[L_tot-sum(L_1,1),LL(1:end-1)];
%% 直接从原来sl得到参数 
h=sl(2*LN+1:2*LN+Ln);
s=sl(3*LN+1+1:3*LN+1+nw);
I_Ln=eye(Ln);
I_nlayer=fliplr(eye(nlayer));
M_realign=kron(I_nlayer,I_Ln);
t=sl(1:LN);
t=t*M_realign;
I_nlayer_1=fliplr(eye(nlayer-1));
M_realign_1=kron(I_nlayer_1,I_Ln);
tdu=sl(3*LN+nw+1+1:end);
tdu=tdu*M_realign_1;
tdu_=[tdu,zeros(1,Ln)];
delta_t=[zeros(1,Ln),tdu];
d=sl(LN+1:2*LN);
d=d*M_realign;
d_end=d(end-Ln+1:end);
LL=sl(2*LN+Ln+1:3*LN+1);
L_1=LL(1:end-1)*M_realign_1;
L_tot=LL(end);
L_down=L_tot-sum(reshape(LL(1:end-1),Ln,nlayer-1).',1);
L=[L_down,L_1];
%%
w=reshape(repmat(s,nh,1),1,nw*nh);
delta=delta*1e-3;
w_tot=sum(s)+(nw+1)*delta;
a=w_tot;
% a=0.05;
% h_tot= sum(h)+(nw-1)*delta;
c=d_end;
% f=linspace(100,6400,200);
x=zeros(1,Ln);
x(1:nh)=s(1)/2;
% x(1:nw)=0;
for i=1:nw-1
    x(nh*i+1:nh*(i+1))=x(nh*i)+s(i)/2+delta+s(i+1)/2;
end
% x=x+deltar;

% x=x-w/2;%角位置x
y=zeros(1,Ln);
for i=1:nw
    for j=2:nh
        y(nh*(i-1)+1)=h(nh*(i-1)+1)/2;
        y(nh*(i-1)+j)=y(nh*(i-1)+j-1)+h(nh*(i-1)+j-1)/2+delta+h(nh*(i-1)+j)/2;
    end
end
% y=y+deltar;
%% complex equivalent parameters
sen=d.^2;%aera of embedded neck,se(se1,se2,...)
sc=w.*h;%aera of cavity
dsc=repmat(sc,1,nlayer)-(d+2*delta).^2;%aera of delta_sc

% cor=enddelta_fit(t,d,repmat(w,1,nlayer),repmat(h,1,nlayer));%end correction
% cor=enddelta_fit(t+delta_t,d,repmat(w,1,nlayer),repmat(h,1,nlayer));%end correction
cor=enddelta_fit(t-delta,d,repmat(w,1,nlayer),repmat(h,1,nlayer));%end correction
cor_=enddelta_fit(tdu_,d,[repmat(w,1,nlayer-1),a*ones(1,Ln)],[repmat(h,1,nlayer-1),a*ones(1,Ln)]);%end correction
[cc0,rhoc0,kc0]=fun_kc_square_comsol([w,repmat(w,1,nlayer)-d,d],[h,repmat(h,1,nlayer)-d,d],f);%complex equivalent parameters of square tube
cc=cc0(:,1:Ln);rhoc=rhoc0(:,1:Ln);kc=kc0(:,1:Ln);%complex equivalent parameters of cavity
ccd=cc0(:,Ln+1:Ln+LN);rhocd=rhoc0(:,Ln+1:Ln+LN);kcd=kc0(:,Ln+1:Ln+LN);%complex equivalent parameters of cavity_delta
cen=cc0(:,Ln+LN+1:Ln+2*LN);rhoen=rhoc0(:,Ln+LN+1:Ln+2*LN);ken=kc0(:,Ln+LN+1:Ln+2*LN);%complex equivalent parameters of embedded neck
%% calculations of impedance
% Zc0=cc0.*rhoc0;
Zc=cc.*rhoc;
Zcd=ccd.*rhocd;
Zcd_down=Zcd;%Z of downward embedded-neck 
Zcd_up=[Zcd(:,LN-Ln+1:LN),Zcd(:,1:LN-Ln)];%Z of upward embedded-neck 
kcd_down=kcd;
kcd_up=[kcd(:,LN-Ln+1:LN),kcd(:,1:LN-Ln)];
Zen=cen.*rhoen;
dsc_down=dsc;%aera of downward cavity_delta
dsc_up=[sc,dsc(1:LN-Ln)];%aera of upward cavity_delta
fL=length(f);
zHR=ones(fL,Ln).*cot(delta_t(1:Ln));
lc=L-delta_t-t+delta;
delta_t=[delta_t,zeros(1,Ln)];
for n=1:nlayer
    zcd_up=-1j*Zcd_up(:,(n-1)*Ln+1:n*Ln).*cot(kcd_up(:,(n-1)*Ln+1:n*Ln).*delta_t((n-1)*Ln+1:n*Ln))./dsc_up((n-1)*Ln+1:n*Ln);
    zc_down=1./(1./zcd_up+1./zHR);
%     lc=(L((n-1)*Ln+1:n*Ln)-delta_t((n-1)*Ln+1:n*Ln)-t((n-1)*Ln+1:n*Ln))+delta;
    kclc=kc.*lc((n-1)*Ln+1:n*Ln);
    Zc_sc=Zc./sc;
    zc_up=-1j*cot(kclc).*Zc_sc.*(1+1j*tan(kclc).*Zc_sc./zc_down)./(1-1j*cot(kclc).*Zc_sc./zc_down);
    zcd_down=-1j*Zcd_down(:,(n-1)*Ln+1:n*Ln).*cot(kcd_down(:,(n-1)*Ln+1:n*Ln).*(t((n-1)*Ln+1:n*Ln)-delta))./dsc_down((n-1)*Ln+1:n*Ln);
    zc=1./(1./zc_up+1./zcd_down);
    zc=zc+1j*omega'*rho0.*cor((n-1)*Ln+1:n*Ln)./sen((n-1)*Ln+1:n*Ln);
    zc=zc+sqrt(2*omega'*u*rho0)./sen((n-1)*Ln+1:n*Ln);%The friction resistance
    kent=ken(:,(n-1)*Ln+1:n*Ln).*(t((n-1)*Ln+1:n*Ln)+delta_t((n)*Ln+1:(n+1)*Ln));
    Zen_sen=Zen(:,(n-1)*Ln+1:n*Ln)./sen((n-1)*Ln+1:n*Ln);
    % zHR=-1j*cot(kent).*Zen_sen.*(1+1j*tan(kent).*Zen_sen./zc)./(1-1j*cot(kent).*Zen_sen./zc);%wrong%%Source expression
    zHR=Zen_sen.*(zc+1j*Zen_sen.*tan(kent))./(Zen_sen+1j*zc.*tan(kent));
    zHR=zHR+1.0j*omega'*rho0.*cor_((n-1)*Ln+1:n*Ln)./sen((n-1)*Ln+1:n*Ln);
    % zHR=zHR+0.3*sqrt(2*omega'*u*rho0)./sen((n-1)*Ln+1:n*Ln);%The friction resistance
    zHR=zHR+1*sqrt(2*omega'*u*rho0)./sen((n-1)*Ln+1:n*Ln);%The friction resistance
end
sen_end=sen((nlayer-1)*Ln+1:nlayer*Ln);
zHR=zHR-1j*omega'*rho0.*cor_((nlayer-1)*Ln+1:nlayer*Ln)./sen_end;

if mnum==200
    zHR=zHR+(0.022131738+1i*k'*0.000306564)*z00./sen_end;
elseif mnum==300
    zHR=zHR+(0.040097229+1i*k'*0.000277)*z00./sen_end;
elseif mnum==500
    zHR=zHR+(0.081062168+1i*k'*0.000193)*z00./sen_end;
elseif mnum==1000
    zHR=zHR+(0.115835878+1i*k'*0.000479105)*z00./sen_end;
elseif mnum==2300
    zHR=zHR+(0.210614769+1i*k'*0.000930718)*z00./sen_end;
elseif mnum==2800
    zHR=zHR+(0.244760911+1i*k'*0.001491064)*z00./sen_end;
elseif mnum==250
    zHR=zHR+(0.032098614+1i*k'*0.000298567)*z00./sen_end;
elseif mnum==150
    zHR=zHR+(0.012185737+1i*k'*0.000338435)*z00./sen_end;
elseif mnum==100
    zHR=zHR+(0.006002611+1i*k'*0.00042147)*z00./sen_end;
elseif mnum==50
    zHR=zHR+(0.000595122+1i*k'*0.000608199)*z00./sen_end;
elseif mnum==10
    zHR=zHR+(0.002929836+1i*k'*0.001449882)*z00./sen_end;        
end
zs=zHR.*sen_end;
end
