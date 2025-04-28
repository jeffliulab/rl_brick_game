# test_env.py
import gymnasium as gym

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=4)
obs, info = env.reset()
print("加载成功，obs shape =", obs.shape)
env.close()

# test cuda
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.get_device_name(0))