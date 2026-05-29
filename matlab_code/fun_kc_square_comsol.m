%%%% Nengyin WANG 2021.4.17 与comsol对比波矢，修改fun_kc_square%%%%
function [cc,rhoc,kc]=fun_kc_square_comsol(w0,h0,f)
gamma0 = 1.4;
c=343.20;
omega0 = 2*pi*f;
% k0 = omega0/c;
rho=1.2043;
Cp=1005.4;
% kappa=0.0267;
kappa=0.025685;%NEHR
% kappa=0.025768;%Groove
mu=1.814*10^-5;
kv2 = -1j*omega0'*rho/mu;
kh2 = -1j*omega0'*rho*Cp/kappa;
kj2=[kv2,kh2];%j=1,2;1=v,2=h,下同
% kv=sqrt(kv2);
% kh=sqrt(kh2);
MMmax=100;
a=w0/2;
b=h0/2;
% alphav=zeros(MMmax,1);
% betav=zeros(MMmax,1);
% alphah=zeros(MMmax,1);
% betav=zeros(MMmax,1);
fL=length(f);
wL=length(w0);
alphaj=zeros(fL,wL,MMmax,2);
betaj=zeros(fL,wL,MMmax,2);
psijm=zeros(fL,wL,MMmax,2);
for j=1:1:2
    for m=1:1:MMmax
        mm=(m-1/2)*pi;
%         alphaj(m,j)=sqrt(kj2(j)-(mm./a).^2);
%         betaj(m,j)=sqrt(kj2(j)-(mm./b).^2);
%         psijm(m,j)=kj2(j).*(alphaj(m,j).*mm).^(-2).*(1-tan(alphaj(m,j).*b)./(alphaj(m,j).*b))+(betaj(m,j).*mm).^(-2).*(1-tan(betaj(m,j).*a)./(betaj(m,j).*a));
        alphaj(:,:,m,j)=sqrt(kj2(:,j)-(mm./a).^2);
        betaj(:,:,m,j)=sqrt(kj2(:,j)-(mm./b).^2);
        psijm(:,:,m,j)=kj2(:,j).*((alphaj(:,:,m,j).*mm).^(-2).*(1-tan(alphaj(:,:,m,j).*b)./(alphaj(:,:,m,j).*b))+(betaj(:,:,m,j).*mm).^(-2).*(1-tan(betaj(:,:,m,j).*a)./(betaj(:,:,m,j).*a)));
    end
end
psij=sum(psijm,3);
cc=c.*sqrt(psij(:,:,1,1)./(gamma0-(gamma0-1).*psij(:,:,1,2)));
rhoc=rho./psij(:,:,1,1);
kc=omega0'./cc;
end
%%测试
% clear;
% w0=[0.0024798,0.0026967];
% h0=[0.0029189,0.0050306];
% f=[3430,3440,3450];
% gamma0 = 1.4;
% c=343.20;
% omega0 = 2*pi*f;k0 = omega0/c;
% rho=1.2043;
% Cp=1005.4;
% % kappa=0.0267;
% % kappa=0.026;
% kappa=0.025675;
% 
% mu=1.814*10^-5;
% kv2 = -1j*omega0'*rho/mu;
% kh2 = -1j*omega0'*rho*Cp/kappa;
% kj2=[kv2,kh2];%j=1,2;1=v,2=h,下同
% % kv=sqrt(kv2);
% % kh=sqrt(kh2);
% MMmax=100;
% a=w0/2;
% b=h0/2;
% % alphav=zeros(MMmax,1);
% % betav=zeros(MMmax,1);
% % alphah=zeros(MMmax,1);
% % betav=zeros(MMmax,1);
% % alphaj=zeros(MMmax,2);
% % betaj=zeros(MMmax,2);
% % psijm=zeros(MMmax,2);
% for j=1:1:2
%     for m=1:1:MMmax
%         mm=(m-1/2)*pi;
% %         alphaj(m,j)=sqrt(kj2(j)-(mm./a).^2);
% %         betaj(m,j)=sqrt(kj2(j)-(mm./b).^2);
% %         psijm(m,j)=kj2(j).*(alphaj(m,j).*mm).^(-2).*(1-tan(alphaj(m,j).*b)./(alphaj(m,j).*b))+(betaj(m,j).*mm).^(-2).*(1-tan(betaj(m,j).*a)./(betaj(m,j).*a));
%         alphaj(:,:,m,j)=sqrt(kj2(:,j)-(mm./a).^2);
%         betaj(:,:,m,j)=sqrt(kj2(:,j)-(mm./b).^2);
%         psijm(:,:,m,j)=kj2(:,j).*((alphaj(:,:,m,j).*mm).^(-2).*(1-tan(alphaj(:,:,m,j).*b)./(alphaj(:,:,m,j).*b))+(betaj(:,:,m,j).*mm).^(-2).*(1-tan(betaj(:,:,m,j).*a)./(betaj(:,:,m,j).*a)));
%     end
% end
% psij=sum(psijm,3);
% cc=c.*sqrt(psij(:,:,1,1)./(gamma0-(gamma0-1).*psij(:,:,1,2)));
% rhoc=rho./psij(:,:,1,1);
% kc=omega0'./cc;