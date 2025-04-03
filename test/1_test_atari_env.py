import gymnasium as gym
import ale_py

# 打印版本信息
print(f"Gymnasium version: {gym.__version__}")
print(f"ALE-Py version: {ale_py.__version__}")

# 尝试加载 Breakout-v5 环境
try:
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    obs, info = env.reset()
    print(f"环境加载成功！Observation shape: {obs.shape}")
    env.close()
except Exception as e:
    print(f"加载环境时出错：{e}")

