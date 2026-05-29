% 端部修正 end correction
% 输入：t,d,size1,size2
% 输出：delta 端部修正量
% 重点注意：MATLAB的if-elseif-else语句是离散不可微的控制流，
% 生成的数据也许会对PyTorch的自动微分产生影响

function [ delta ] = enddelta_fit( t,d,size1,size2 )
kesi=d./min(size1,size2);
a=2/3;b=-26/15;c=16/15;A=pi*d.^2/4;
    if kesi<0.4&t<3e-3
       delta=0.425*d.*(1-1.25.*kesi);
    elseif kesi<0.4
         delta=0.3*d.*(1-1.25.*kesi);
    else
        delta=sqrt(A).*(a*kesi.^2+b.*kesi+c)/2;
    end
end

