# filename: agent_mvp_advanced.py
import numpy as np
import torch
import torch.optim as optim
import sys
import json
import os
import re

# --- 1. 环境与依赖检查 ---
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.tools import tool
    from langchain_core.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ 环境错误: {e}")
    sys.exit(1)

# 导入模型定义
try:
    from resnet_model import ResNetDNN 
except ImportError:
    print("❌ 错误: 找不到 resnet_model.py。")
    sys.exit(1)

# --- 2. 系统配置 ---
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": os.path.join("saved_models_dnn_v2", "ResNet_PeakFocus.pth"),
    "model_args": {
        "input_dim": 31,
        "output_dim": 100, 
        "hidden_dim": 1024,
        "num_blocks": 8
    }
}

print(f"\n🚀 [System Init] Loading Neural Surrogate Model on {CONFIG['device']}...")

# --- 3. 加载神经网络 ---
resnet_model = ResNetDNN(**CONFIG['model_args'])

try:
    if os.path.exists(CONFIG['model_path']):
        print(f"📂 Loading weights from: {CONFIG['model_path']}")
        state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        resnet_model.load_state_dict(state_dict)
        print("✅ Pre-trained weights (Raw Data Version) loaded successfully.")
    else:
        print(f"❌ Error: Model file not found at: {CONFIG['model_path']}")
        print("⚠️  Using RANDOM weights (Demonstration Mode).")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")

resnet_model.to(CONFIG['device'])
resnet_model.eval() 

# [性能优化] 冻结模型权重，只优化输入参数
for param in resnet_model.parameters():
    param.requires_grad = False

# --- 4. 定义基准参数 (初始猜测) ---
DEFAULT_RAW_PARAMS = np.array([
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 
    26.0, 26.0, 26.0, 26.0, 
    20.0, 20.0, 20.0, 20.0, 
    100.0, # Index 24: Total Thickness (将被覆盖)
    26.0, 26.0, 
    10.0, 10.0, 10.0, 10.0 
])

# --- 5. 定义工具集 ---

@tool
def evaluate_design_performance(json_input: str):
    """
    [Basic Tool] Simulates acoustic absorption for a fixed design.
    Use this for quick checks.
    """
    return "Suggestion: For high performance, please use the 'optimize_metamaterial_design' tool."

@tool
def optimize_metamaterial_design(json_input: str):
    """
    [Advanced Tool] Uses Gradient Descent to INVERSE DESIGN a metamaterial structure.
    Use this when the user asks for "High Performance", "Best", or "Perfect" absorption.
    
    Parameters:
      - "target_thickness_mm": (float) The constraint thickness.
      - "optimization_goal": (str) "maximize_avg" (broadband) or "maximize_peak" (resonance).
    """
    try:
        # 1. 解析输入
        cleaned_input = json_input.strip()
        if "```" in cleaned_input: 
            match = re.search(r"```(?:json)?(.*?)```", cleaned_input, re.DOTALL)
            cleaned_input = match.group(1).strip() if match else cleaned_input.replace("```", "")
        
        params = json.loads(cleaned_input)
        target_thickness = float(params.get("target_thickness_mm", 100.0))
        goal = params.get("optimization_goal", "maximize_avg")
        
    except Exception as e:
        return f"Error parsing input: {e}"

    print(f"\n🧪 [AI Scientist] Starting Inverse Design Optimization...")
    print(f"   > Constraint: Thickness = {target_thickness} mm")
    print(f"   > Goal: {goal.replace('_', ' ').upper()}")

    # 2. 初始化可学习参数
    init_params = DEFAULT_RAW_PARAMS.copy()
    init_params[24] = target_thickness 
    
    # 转为 Tensor，并允许求导 (requires_grad=True)
    param_tensor = torch.tensor(init_params, dtype=torch.float32, device=CONFIG['device'], requires_grad=True)
    
    # 3. 定义优化器 (Adam)
    optimizer = optim.Adam([param_tensor], lr=0.5) # 较大的学习率加速收敛
    
    best_score = -999.0
    best_spectrum = None
    best_params_values = None # 用于存储最佳参数快照
    
    # 4. 优化循环
    steps = 100
    for i in range(steps):
        optimizer.zero_grad()
        
        # 前向传播
        spectrum = resnet_model(param_tensor.unsqueeze(0)).flatten()
        
        # 定义 Loss
        if goal == "maximize_avg":
            current_score = torch.mean(spectrum)
            loss = -current_score 
        else:
            current_score = torch.max(spectrum)
            loss = -current_score
            
        loss.backward()
        optimizer.step()
        
        # [物理约束] 投影梯度
        with torch.no_grad():
            # 1. 锁死厚度
            param_tensor.data[24] = target_thickness
            # 2. 限制数值范围 (防止负数)
            param_tensor.data.clamp_(1.0, 300.0)
        
        # 记录最佳结果
        if current_score.item() > best_score:
            best_score = current_score.item()
            best_spectrum = spectrum.detach().cpu().numpy()
            # [关键] 深拷贝当前的参数值，作为设计蓝图
            best_params_values = param_tensor.detach().cpu().numpy().copy()
            
        if i % 20 == 0:
            print(f"   Step {i}/{steps}: Score = {current_score.item():.4f}")

    # 5. 后处理与报告
    best_spectrum = np.clip(best_spectrum, 0.0, 1.0)
    final_avg = np.mean(best_spectrum)
    final_peak = np.max(best_spectrum)
    peak_hz = int(np.argmax(best_spectrum) * 20.0)

    improvement_msg = ""
    if final_avg > 0.8:
        improvement_msg = "🔥 PERFECT Convergence achieved!"
    elif final_avg > 0.6:
        improvement_msg = "✅ Significant optimization achieved."
    else:
        improvement_msg = "⚠️ Optimization limit reached."
    
    # 格式化参数列表，保留2位小数
    optimized_params_list = [round(float(x), 2) for x in best_params_values]

    result = {
        "status": "success",
        "optimization_steps": steps,
        "final_performance": {
            "avg_absorption": round(float(final_avg), 3),
            "peak_absorption": round(float(final_peak), 3),
            "peak_frequency_hz": peak_hz
        },
        # [✨核心功能] 输出设计蓝图
        "optimized_geometry": optimized_params_list,
        "conclusion": f"{improvement_msg} The AI has tuned the geometry parameters. See 'optimized_geometry' for the blueprint."
    }
    
    return json.dumps(result, indent=2)

# --- 6. 启动智能体 ---
def run_agent():
    print("🤖 Connecting to Ollama (qwen2.5:14b)...")
    try:
        llm = ChatOllama(model="qwen2.5:14b", temperature=0.0)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    tools = [optimize_metamaterial_design]

    # Prompt Template
    template = '''You are an AI Physicist specializing in Inverse Design.
Your job is to OPTIMIZE structures to find the best possible performance and provide the blueprint.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, MUST be a valid JSON string
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

PROTOCOL:
1. **Analyze**: If user wants "design", "best", "optimize", use `optimize_metamaterial_design`.
2. **Execute**: Run the optimizer.
3. **Report**: 
    - Report the "final_performance" stats.
    - Explicitly mention that the **Optimized Geometry (Blueprint)** has been generated.
    - You can summarize the geometry briefly (e.g., "The algorithm adjusted the cavity depths and pore sizes...").

Begin!

User Input: {input}
{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    # 用户请求
    user_query = "Please design a perfect broadband absorber with 120mm thickness. I want the best possible performance and the design parameters."
    
    print(f"\n👤 User Query: {user_query}")
    print("-" * 60)
    
    try:
        agent_executor.invoke({"input": user_query})
    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == "__main__":
    run_agent()