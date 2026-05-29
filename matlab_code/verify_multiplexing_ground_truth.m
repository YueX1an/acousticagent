% verify_multiplexing_64x64_gt.m
% 用于验证 Python 检索出的 64x64 多频复用物理参数的真实声学响应
% 并自动导出纯数据供 Python 渲染顶刊矢量图

clc; clear; close all;

% ================= 🎛️ 1. 读取 Python 生成的参数 =================
disp('📥 正在读取 PhyAR-Net 检索生成的参数 (matched_params_64x64.txt)...');

% 设定根目录下的输入与输出文件名
input_file = 'matched_params_64x64_weighted.txt';
output_file = '20260329_alpha_GT_64x64.txt';

if ~exist(input_file, 'file')
    error(['❌ 找不到文件: ', input_file, '，请确认它已放置在 MATLAB 的当前工作目录下。']);
end

params = load(input_file); 
[num_pixels, num_dims] = size(params);
grid_size = sqrt(num_pixels); % 应该是 64

if num_dims ~= 31
    error('❌ 参数维度错误！期望 31 维，实际读取为 %d 维', num_dims);
end

% ================= 🎛️ 2. 物理仿真核心配置 =================
% 【严格对齐】必须与您的 generate_31param_data.m 保持绝对一致
nlayer = 2; 
ncell = 4; 
nw = 2;
delta = 0.7; 
mnum = 500;    % 模态截断数
Norder = 9;    % 阶数

% 验证这四个靶点频率
target_freqs = [400, 900, 1400, 1900]; 
fL = length(target_freqs);

% 预分配内存，存放真实的吸声系数 alpha
alpha_ground_truth = zeros(num_pixels, fL);

% ================= 🚀 3. 大规模并行物理仿真 =================
disp(['🚀 开始严格物理仿真验证 (', num2str(num_pixels), ' 个像素点 * 4 个频率)...']);
tic;
% 如果您的电脑开启了并行池，这里将极速运行
parfor i = 1:num_pixels
    sl = params(i, :);
    
    % 调用您最底层的真实物理算子
    [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
        target_freqs, sl, ncell, nw, delta, nlayer, mnum, Norder);
    
    alpha_ground_truth(i, :) = alpha;
end
time_cost = toc;
disp(['✅ 仿真完成！耗时: ', num2str(time_cost), ' 秒']);

% ================= 💾 4. 导出纯数据给 Python =================
disp('💾 正在导出 MATLAB 仿真真值矩阵...');
save(output_file, 'alpha_ground_truth', '-ascii');
disp(['✅ 真值数据已完美导出至根目录: ', output_file]);

% ================= 🎨 5. 绘图验证 (真实验证矩阵) =================
disp('🎨 正在绘制 MATLAB 真值全息图...');
figure('Name', 'Ground Truth Verification 64x64', 'Position', [100, 100, 1400, 350]);
freq_labels = {'400 Hz (Taiji)', '900 Hz (Tongji)', '1400 Hz (Smiley)', '1900 Hz (Heart)'};

% 使用标准的 jet 色标进行快速校验
custom_cmap = jet(256); 

for j = 1:fL
    % 提取当前频率的所有 4096 个像素的吸声率
    img_1d = alpha_ground_truth(:, j);
    
    % 【极其关键的转置】先 reshape 再转置 (.)，还原 Python 的 Row-major 排布
    img_2d = reshape(img_1d, grid_size, grid_size).';
    
    subplot(1, 4, j);
    imagesc(img_2d);
    colormap(custom_cmap);
    caxis([0.0 1.0]); % 【绝对锁定刻度】
    colorbar;
    
    title(freq_labels{j}, 'FontWeight', 'bold', 'FontSize', 12);
    axis off;
    axis square;
    
    % 在图下方标注真实的平均吸声率 (针对 64x64 动态调整了文本的 Y 轴位置)
    mean_abs = mean(img_1d);
    text(grid_size/2, grid_size + 8, sprintf('GT Mean Abs: %.2f', mean_abs), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

% 保存高清真值预览图
preview_img = 'MATLAB_GT_Preview_64x64.png';
saveas(gcf, preview_img);
disp(['💾 验证预览图已保存为 ', preview_img]);
disp('🎉 闭环跑通！现在您可以去运行 Python 的终极矢量画图引擎了！');