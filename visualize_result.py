import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from resnet_model import ResNetDNN

# --- 1. 配置 ---
MODEL_PATH = os.path.join("saved_models_dnn_v2", "ResNet_PeakFocus.pth")
DEVICE = torch.device("cpu") # 画图用 CPU 足够了

# --- 2. 加载模型 ---
model = ResNetDNN(input_dim=31, output_dim=100, hidden_dim=1024, num_blocks=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 3. 输入您的 "完美设计参数" (从 Agent 日志中复制) ---
# 这就是您刚刚得到的 optimized_geometry
design_params = [
    1.0, 1.0, 1.0, 1.0, 18.38, 14.26, 15.73, 9.83, 
    13.89, 12.81, 12.53, 10.31, 4.24, 10.3, 21.72, 12.84, 
    46.28, 40.94, 35.08, 51.96, 42.92, 22.05, 27.99, 23.06, 
    120.0, # Total Thickness
    30.05, 16.39, 13.27, 15.71, 4.64, 14.32
]

# --- 4. 预测频谱 ---
input_tensor = torch.tensor(design_params, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    spectrum = model(input_tensor).flatten().numpy()
    spectrum = np.clip(spectrum, 0.0, 1.0) # 物理截断

# --- 5. 绘图 ---
freqs = np.linspace(0, 2000, 100) # 假设 100个点对应 0-2000Hz

plt.figure(figsize=(10, 6))
plt.plot(freqs, spectrum, 'r-', linewidth=3, label='AI Optimized Design')

# 标记峰值
peak_idx = np.argmax(spectrum)
plt.plot(freqs[peak_idx], spectrum[peak_idx], 'ko')
plt.text(freqs[peak_idx], spectrum[peak_idx]+0.02, 
         f'Peak: {spectrum[peak_idx]:.2f} @ {int(freqs[peak_idx])}Hz', 
         ha='center', fontsize=12)

# 标记平均值
avg_val = np.mean(spectrum)
plt.axhline(y=avg_val, color='b', linestyle='--', label=f'Average: {avg_val:.3f}')

plt.title(f'AI Designed Broadband Absorber (t=120mm)', fontsize=15)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Absorption Coefficient', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.fill_between(freqs, spectrum, alpha=0.1, color='red')

# 保存
save_path = "optimized_spectrum.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Chart saved to {save_path}")
plt.show()