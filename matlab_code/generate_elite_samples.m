% =========================================================================
% generate_elite_samples.m
% 基于宽带吸声精英样本的局部邻域扩增 (零和微扰法，100%防穿模)
% =========================================================================
clc; clear; close all;

% --- 1. 填入您的“黄金”精英样本 (31维) ---
elite_sl = [8.0000 2.6875 1.3750 2.6875 1.0125 13.6531 1.0125 9.1449 ...
            10.9750 10.2250 10.2563 11.2023 21.2588 8.8620 13.4662 5.4523 ...
            25.2588 21.4412 17.4662 29.2338 ... % h1~h4
            49.9580 20.7096 50.0000 31.1604 ... % Lu1~Lu4
            57.0000 ...                         % Ltot
            26.3005 20.3995 ...                 % w1, w2
            0.0112 14.8874 0.1250 15.2180];     % tu1~tu4

% --- 2. 扩增设置 ---
N_elite = 100000;         % 需要生成的精英衍生样本数量
noise_ratio = 0.10;     % 扰动幅度 (10%，即允许在原值上下 10% 范围内波动)
delta = 1.0;            % 壁厚

% 预分配内存
sl_elite_out = zeros(N_elite, 31);
valid_count = 0;

disp(['开始围绕精英样本进行局部扩增，目标生成 ', num2str(N_elite), ' 个...']);

while valid_count < N_elite
    new_sl = elite_sl;
    
    % --- 独立参数微扰 (加上随机噪声) ---
    % 1. 管长 t (1-8): [1, 200]
    for i = 1:8
        fluct = elite_sl(i) * noise_ratio * (2*rand() - 1);
        new_sl(i) = max(1.0, min(200.0, elite_sl(i) + fluct));
    end
    
    % 2. 上层腔长 Lu (21-24): [10, 50]
    for i = 21:24
        fluct = elite_sl(i) * noise_ratio * (2*rand() - 1);
        new_sl(i) = max(10.0, min(50.0, elite_sl(i) + fluct));
    end
    
    % --- Z 轴耦合参数微扰 ---
    % 3. 层间接管 tu (28-31): [0, 200] 且必须遵守 Z轴防穿模
    for i = 1:4
        fluct = elite_sl(27+i) * noise_ratio * (2*rand() - 1);
        tu_candidate = max(0.0, elite_sl(27+i) + fluct);
        % 绝对防穿模：上限受制于当前生成的 Lu
        max_tu = 57.0 - new_sl(20+i) - 3.0; 
        new_sl(27+i) = min(tu_candidate, max_tu);
    end
    
    % --- XY 轴零和微扰 (核心防穿模) ---
    % 4. 高度 h (17-20): 必须满足 h1+h2=46.7, h3+h4=46.7
    % 方法：给 h1 加一段随机值，给 h2 减去同样的随机值
    dh_12 = (elite_sl(17) * noise_ratio) * (2*rand() - 1);
    h1_new = elite_sl(17) + dh_12;
    h2_new = elite_sl(18) - dh_12;
    if h1_new >= 10 && h1_new <= 50 && h2_new >= 10 && h2_new <= 50
        new_sl(17) = h1_new; new_sl(18) = h2_new;
    else
        continue; % 越界则放弃本次微扰
    end
    
    dh_34 = (elite_sl(19) * noise_ratio) * (2*rand() - 1);
    h3_new = elite_sl(19) + dh_34;
    h4_new = elite_sl(20) - dh_34;
    if h3_new >= 10 && h3_new <= 50 && h4_new >= 10 && h4_new <= 50
        new_sl(19) = h3_new; new_sl(20) = h4_new;
    else
        continue; 
    end
    
    % 5. 宽度 w (26-27): 必须满足 w1+w2=46.7
    dw = (elite_sl(26) * noise_ratio) * (2*rand() - 1);
    w1_new = elite_sl(26) + dw;
    w2_new = elite_sl(27) - dw;
    if w1_new >= 15 && w1_new <= 40 && w2_new >= 15 && w2_new <= 40
        new_sl(26) = w1_new; new_sl(27) = w2_new;
    else
        continue; 
    end
    
    % --- 安全孔径生成 ---
    % 6. 孔径 d (9-16): 必须小于所在腔体的 min(w, h) - 4
    d_is_safe = true;
    for i = 1:8
        % 判断当前孔属于左侧还是右侧，上层还是下层
        if i==1 || i==2 || i==5 || i==6; w_current = new_sl(26); else; w_current = new_sl(27); end
        if i==1 || i==5; h_current = new_sl(17); elseif i==2 || i==6; h_current = new_sl(18);
        elseif i==3 || i==7; h_current = new_sl(19); else; h_current = new_sl(20); end
        
        limit_d = min(w_current, h_current) - 4.0;
        
        fluct = elite_sl(8+i) * noise_ratio * (2*rand() - 1);
        d_candidate = max(1.0, elite_sl(8+i) + fluct);
        
        if d_candidate > limit_d
            d_is_safe = false; break;
        end
        new_sl(8+i) = d_candidate;
    end
    
    if ~d_is_safe
        continue; % 孔径干涉，放弃
    end
    
    % 如果所有检查都通过，保存该精英样本
    valid_count = valid_count + 1;
    sl_elite_out(valid_count, :) = new_sl;
    
    if mod(valid_count, 200) == 0
        fprintf('已生成 %d / %d 个精英衍生样本...\n', valid_count, N_elite);
    end
end

disp('精英样本扩增完成！');

% --- 保存为独立数据集 ---
save("elite_slN_1000.txt", "sl_elite_out", "-ascii");
disp('已保存至 elite_slN_1000.txt，您可以直接导入仿真或用于深度学习！');