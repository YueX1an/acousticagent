%% 零空间采样示例：修正约束转换错误
%%%% 王能银 20250502 生成随机的多单元N层腔 零空间采样函数
function [samples_x]=NullSpace_HitandRun_func(Aeq,beq,A,b,lb,ub,num_samples)
rng('shuffle');  % 添加此行，随机种子基于系统时间

% % ========== 步骤1: 定义约束条件 ==========
% Aeq = [1, 1, 1];        % x1 + x2 + x3 = 1
% beq = 1;
% 
% A = [-1, 0, 0;          % x1 ≥ 0.2
%       0, 1, 0];         % x2 ≤ 0.8
% b = [-0.2; 0.8];
% 
% lb = [0.1; 0.1; 0.1];
% ub = [0.9; 0.9; 0.9];

% ========== 步骤2: 计算零空间基和特解 ==========
x0 = pinv(Aeq) * beq;   % 特解 x0 = [1/3; 1/3; 1/3]
F = null(Aeq);           % 零空间基
n_free = size(F, 2);

% ========== 步骤3: 正确转换约束到自由变量空间 ==========
% 不等式约束 A*(x0 + F*y) <= b
if size(A)==0
    A_ineq_y=[];
    b_ineq_y=[];
else
    A_ineq_y = A * F;
    b_ineq_y = b - A * x0;
end

% 下界约束：F*y >= lb - x0 → -F*y <= -(lb - x0)
A_lb_y = -F;
b_lb_y = -(lb - x0);

% 上界约束：F*y <= ub - x0
A_ub_y = F;
b_ub_y = ub - x0;

% 合并约束
A_combined = [A_ineq_y; A_lb_y; A_ub_y];
b_combined = [b_ineq_y; b_lb_y; b_ub_y];

% ========== 步骤4: 验证原始解可行性 ==========
% x_test = [1/3; 1/3; 1/3];
% y_test = F \ (x_test - x0);  % y_test = [0; 0]

% 检查约束是否满足
% if all(A_combined * y_test <= b_combined + 1e-6)
%     fprintf('原始解在自由变量空间中可行\n');
% else
%     error('约束转换错误，请检查 A_combined 和 b_combined');
% end

% ========== 步骤4: 自动寻找初始可行点 ==========
options = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'dual-simplex');
max_retries = 50;  % 最大重试次数
found = false;

% 定义自由变量上下界（无显式限制）
lb_y = -inf(n_free, 1);
ub_y = inf(n_free, 1);

for retry = 1:max_retries
    % 生成随机目标函数以寻找不同初始点
    % f = randn(n_free, 1);  
    f = randn(n_free, 1) + 0.1*randn(n_free, 1)*retry;  % 增强随机性
    
    % 调用linprog寻找可行点
    [y_test, ~, exitflag] = linprog(f, A_combined, b_combined, [], [], lb_y, ub_y, options);
    
    if exitflag == 1
        found = true;
        % fprintf('成功找到初始点 (尝试次数: %d)\n', retry);
        break;
    end
end

if ~found
    error('无法找到初始可行点，请检查约束是否自洽');
end

% ========== 步骤5: Hit-and-Run 采样 ==========
% num_samples = 10;
samples_y = zeros(n_free, num_samples);
samples_y(:,1) = y_test;  % 初始点

for i = 2:num_samples
    d = randn(n_free, 1);
    d = d / norm(d);
    
    numerator = b_combined - A_combined * samples_y(:,i-1);
    denominator = A_combined * d;
    
    t_min = -inf;
    t_max = inf;
    for j = 1:length(numerator)
        if denominator(j) > 0
            t_max = min(t_max, numerator(j)/denominator(j));
        elseif denominator(j) < 0
            t_min = max(t_min, numerator(j)/denominator(j));
        end
    end
    
    if t_min < t_max
        t = t_min + (t_max - t_min) * rand();
        samples_y(:,i) = samples_y(:,i-1) + t * d;
    else
        samples_y(:,i) = samples_y(:,i-1);
    end
end

% ========== 步骤6: 转换并验证结果 ==========
samples_x = x0 + F * samples_y;
