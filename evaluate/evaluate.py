#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to load and test a trained DQN model on Atari Breakout.
"""

import os
import sys
import numpy as np
import torch
import cv2
import time
from collections import deque
import gymnasium as gym


# 改进的模型路径查找
def find_model_path(model_filename="model_4_2M.pt"):
    # 1. 检查当前目录
    if os.path.exists(model_filename):
        return model_filename
    
    # 2. 检查脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_filename)
    if os.path.exists(model_path):
        return model_path
    
    # 3. 检查上级目录
    parent_dir = os.path.dirname(script_dir)
    model_path = os.path.join(parent_dir, model_filename)
    if os.path.exists(model_path):
        return model_path
    
    # 4. 检查特定子目录
    for subdir in ['models', 'output', 'checkpoints']:
        model_path = os.path.join(script_dir, subdir, model_filename)
        if os.path.exists(model_path):
            return model_path
        
        model_path = os.path.join(parent_dir, subdir, model_filename)
        if os.path.exists(model_path):
            return model_path
    
    # 未找到模型文件
    print(f"错误: 无法找到模型文件 '{model_filename}'")
    print("已搜索以下路径:")
    print(f" - 当前目录: {os.getcwd()}")
    print(f" - 脚本目录: {script_dir}")
    print(f" - 上级目录: {parent_dir}")
    print("当前目录文件列表:")
    print(os.listdir("."))
    return None

# 检查并安装必要的依赖
def check_dependencies():
    try:
        import gymnasium as gym
        try:
            import ale_py
            print("ALE-Py 已安装")
        except ImportError:
            print("缺少 ALE-Py 包，尝试安装...")
            os.system("pip install ale-py")
            try:
                import ale_py
                print("ALE-Py 安装成功!")
            except ImportError:
                print("ALE-Py 安装失败，请手动运行: pip install ale-py")
                
        # 尝试加载Atari环境
        try:
            env = gym.make("ALE/Breakout-v5")
            env.close()
            print("Atari环境测试成功!")
        except Exception as e:
            if "Namespace ALE not found" in str(e):
                print("缺少Atari ROM许可证，尝试安装...")
                os.system('pip install "gymnasium[accept-rom-license]"')
                try:
                    env = gym.make("ALE/Breakout-v5")
                    env.close()
                    print("Atari环境现在可以正常加载!")
                except Exception as e2:
                    print(f"尝试安装后仍然失败: {e2}")
                    print("请手动运行以下命令:")
                    print('pip install gymnasium[atari]')
                    print('pip install "gymnasium[accept-rom-license]"')
            else:
                print(f"无法加载Atari环境: {e}")
                
    except ImportError:
        print("缺少gymnasium包，尝试安装...")
        os.system("pip install gymnasium")
        os.system('pip install "gymnasium[atari]"')
        os.system('pip install "gymnasium[accept-rom-license]"')
        print("请重新运行脚本以确认安装成功")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== Environment Wrappers ==========

class AutoFireResetEnv(gym.Wrapper):
    """开局自动发球的环境包装器"""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, done, truncated, info = self.env.step(1)  # action=1 => FIRE
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class PreprocessAtari(gym.ObservationWrapper):
    """将彩色Atari图像转换为灰度[84x84]，并归一化到[0,1]"""
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.img_size[0], self.img_size[1], 1),
            dtype=np.float32
        )

    def observation(self, obs):
        # 裁剪掉画面上下无用区域
        obs = obs[34:-16, :, :]
        # 缩放至 (84, 84)
        obs = cv2.resize(obs, self.img_size)
        # 转为灰度图 (单通道)
        obs = obs.mean(axis=-1, keepdims=True)
        # 归一化至 [0,1]
        obs = obs.astype(np.float32) / 255.0
        return obs


class FrameBuffer(gym.Wrapper):
    """维护n_frames帧的缓冲区，沿通道维度堆叠"""
    def __init__(self, env, n_frames=4, dim_order='pytorch'):
        super().__init__(env)
        self.dim_order = dim_order
        height, width, channels = env.observation_space.shape

        if dim_order == 'pytorch':
            # PyTorch shape: (C, H, W)
            self.frame_shape = (channels, height, width)
            obs_shape = (channels * n_frames, height, width)
        else:
            raise ValueError('dim_order must be "pytorch" for this code.')

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
        self.n_frames = n_frames
        self.framebuffer = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(obs)
        return self.framebuffer, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or truncated
        self.update_buffer(obs)
        return self.framebuffer, reward, done, truncated, info

    def update_buffer(self, obs):
        # 将[H,W,C]的图像转为[C,H,W]并拼接到framebuffer最前面，移除最旧的一帧
        obs = np.transpose(obs, (2, 0, 1))
        old = self.framebuffer[:-self.frame_shape[0], :, :]
        self.framebuffer = np.concatenate([obs, old])


# ========== DQN Network Definition ==========

class DQN(torch.nn.Module):
    """深度Q网络，输出每个动作的Q(s,a)值"""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        # 计算卷积输出大小
        conv_out_size = self._get_conv_out(input_shape)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


# ========== Agent Definition ==========

class DQNAgent:
    """DQN智能体，使用策略网络进行决策"""
    def __init__(self, state_shape, n_actions, epsilon=0.0):
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.policy_net = DQN(state_shape, n_actions).to(device)

    def get_qvalues(self, states):
        """计算状态的Q值"""
        with torch.no_grad():
            if isinstance(states, np.ndarray):
                states = torch.from_numpy(states).float()
            states = states.to(device)
            qvals = self.policy_net(states)
            return qvals.cpu().numpy()

    def sample_actions(self, qvalues):
        """根据Q值选择动作"""
        batch_size = qvalues.shape[0]
        random_actions = np.random.choice(self.n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=1)
        explore = np.random.rand(batch_size) < self.epsilon
        return np.where(explore, random_actions, best_actions)


# ========== Main Function ==========

def main():
    """主函数：加载模型并运行游戏"""
    # 首先检查依赖
    try:
        import gymnasium as gym
        import ale_py
    except ImportError:
        print("缺少必要的库，正在检查并安装依赖...")
        check_dependencies()
        print("请重新运行脚本")
        sys.exit(1)

    # 查找模型文件
    MODEL_PATH = find_model_path()
    if MODEL_PATH is None:
        sys.exit(1)
    else:
        print(f"找到模型文件: {MODEL_PATH}")
        
    print("正在初始化环境...")
    
    # 初始化环境，使用human模式以便在屏幕上显示游戏
    try:
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        env = AutoFireResetEnv(env)
        env = PreprocessAtari(env)
        env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    except Exception as e:
        print(f"初始化环境失败: {e}")
        print("尝试安装必要的依赖...")
        check_dependencies()
        print("请重新运行脚本")
        sys.exit(1)
        
    print("环境初始化成功!")
    
    # 初始化智能体
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    agent = DQNAgent(state_shape, n_actions, epsilon=0.0)  # 纯贪婪策略
    
    # 加载模型
    print(f"正在加载模型 '{MODEL_PATH}'...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # 尝试不同的键名，因为保存格式可能有所不同
        if 'policy_net_state_dict' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        elif 'state_dict' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
        else:
            # 假设checkpoint直接是state_dict
            agent.policy_net.load_state_dict(checkpoint)
            
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)
    
    # 开始游戏
    print("开始游戏测试...")
    
    episodes = 5  # 玩5局游戏
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        lives = info.get('lives', 5)
        
        print(f"第 {episode+1}/{episodes} 局开始...")
        
        step = 0
        while True:
            # 获取动作
            state_batch = np.array([obs])
            qvals = agent.get_qvalues(state_batch)
            action = qvals.argmax(axis=1)[0]  # 总是选择Q值最高的动作
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # 检查是否失去生命，并显示当前状态
            current_lives = info.get('lives', 0)
            if current_lives < lives:
                lives = current_lives
                print(f"  失去一条生命! 剩余生命: {lives}")
            
            # 每10步显示一次当前分数
            if step % 10 == 0:
                print(f"  步骤 {step}, 当前分数: {episode_reward}")
            
            # 如果游戏结束则跳出循环
            if done or truncated:
                break
            
            # 添加短暂延迟，使游戏速度适合观看
            time.sleep(0.01)
        
        print(f"第 {episode+1} 局结束，总分: {episode_reward}")
    
    env.close()
    print("测试完成!")

if __name__ == "__main__":
    main()