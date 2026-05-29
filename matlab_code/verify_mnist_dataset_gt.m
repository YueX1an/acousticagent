% verify_mnist_dataset_gt.m
% 用于验证 Python 检索出的 MNIST (1907) 物理参数的真实声学响应

clc; clear; close all;

% ================= 🎛️ 1. 核心配置与路径 =================
grid_size = 15;
target_freqs = [400, 900, 1400, 1900]; % 对应 1, 9, 0, 7 的四个频段
fL = length(target_freqs);

% 文件夹与文件路径
data_dir = 'txt_dataset_matching_mnist/';
input_file = fullfile(data_dir, 'matched_params_mnist.txt');
output_file = fullfile(data_dir, 'alpha_GT_mnist.txt');

% 物理算子参数 (必须与生成数据集时绝对一致)
nlayer = 2; ncell = 4; nw = 2; delta = 0.7; 
mnum = 500; Norder = 9;    

% ================= 📥 2. 读取参数 =================
disp('📥 正在读取 MNIST 匹配的物理参数...');
if ~exist(input_file, 'file')
    error(['找不到文件: ', input_file, '，请确认 Python 脚本是否正常生成。']);
end
params = load(input_file); 
[num_pixels, num_dims] = size(params);

if num_pixels ~= grid_size^2
    warning('警告: 读取的参数行数 (%d) 与 grid_size^2 (%d) 不匹配！', num_pixels, grid_size^2);
end

% ================= 🚀 3. 大规模并行物理仿真 =================
alpha_ground_truth = zeros(num_pixels, fL);

disp(['🚀 开始极速物理仿真验证 (', num2str(num_pixels), ' 个单元)...']);
tic;
parfor i = 1:num_pixels
    sl = params(i, :);
    [alpha, ~] = fun_structure_broadband_nlayer_HB_func(...
        target_freqs, sl, ncell, nw, delta, nlayer, mnum, Norder);
    alpha_ground_truth(i, :) = alpha;
end
time_cost = toc;
disp(['✅ 仿真完成！耗时: ', num2str(time_cost), ' 秒']);

% ================= 💾 4. 导出纯数据给 Python =================
disp('💾 正在导出 MATLAB 仿真真值矩阵...');
save(output_file, 'alpha_ground_truth', '-ascii');
disp(['✅ 真值数据已导出至: ', output_file]);

% ================= 🎨 5. 快速预览绘图 =================
figure('Name', 'MNIST 1907 Ground Truth', 'Position', [100, 100, 1400, 350]);
freq_labels = {'400 Hz (Digit 1)', '900 Hz (Digit 9)', '1400 Hz (Digit 0)', '1900 Hz (Digit 7)'};
custom_cmap = jet(256); 

for j = 1:fL
    img_1d = alpha_ground_truth(:, j);
    % 关键转置：还原 Python 的 Row-major 逻辑
    img_2d = reshape(img_1d, grid_size, grid_size).';
    
    subplot(1, 4, j);
    imagesc(img_2d); 
    colormap(custom_cmap); 
    caxis([0.0 1.0]); % 绝对锁定刻度
    colorbar; 
    axis off; 
    axis square;
    title(freq_labels{j}, 'FontWeight', 'bold', 'FontSize', 12);
    
    % 在图下方标注真实的平均吸声率
    mean_abs = mean(img_1d);
    text(grid_size/2, grid_size + 3, sprintf('GT Mean Abs: %.2f', mean_abs), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

preview_path = fullfile(data_dir, 'MATLAB_MNIST_GT_Preview.png');
saveas(gcf, preview_path);
disp(['📊 预览图已保存至: ', preview_path]);
disp('🎉 闭环跑通！现在您可以运行 Python 终极矢量画图引擎了！');