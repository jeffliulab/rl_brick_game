#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Atari环境模块 - 用于DQN训练的Atari游戏环境设置

包含环境初始化、预处理和帧缓冲等功能
"""

import sys
import numpy as np
import gymnasium as gym
import ale_py
import cv2

class AutoFireResetEnv(gym.Wrapper):
    """
    在环境重置后自动发射(FIRE)的包装器。
    
    在Breakout等游戏中，需要在开局按FIRE按钮开始游戏。
    这个包装器使得智能体不需要学习"开局发球"这一动作。
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, done, truncated, info = self.env.step(1)  # action=1 => FIRE
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class PreprocessAtari(gym.ObservationWrapper):
    """
    将彩色Atari图像转换为灰度图 [84x84]，并归一化到 [0,1]。
    
    这个预处理步骤是DQN算法的标准操作，可以减少状态空间的维度，
    并使得神经网络更容易学习。
    """
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)

        # 修改观察空间为 (84, 84, 1) 浮点类型
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
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
    """
    维护n_frames帧的缓冲区，沿通道维度堆叠它们。
    
    在DQN中，我们通常使用连续的4帧作为输入，以便网络能够捕捉到
    游戏中的运动信息。dim_order='pytorch' 表示输出形状为 (C, H, W)。
    """
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
            low=0.0,
            high=1.0,
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
        return self.framebuffer, reward, done, info

    def update_buffer(self, obs):
        # 将[H,W,C]的图像转为[C,H,W]并拼接到framebuffer最前面，移除最旧的一帧
        obs = np.transpose(obs, (2, 0, 1))
        old = self.framebuffer[:-self.frame_shape[0], :, :]
        self.framebuffer = np.concatenate([obs, old])


def initialize_environment(game_name="ALE/Breakout-v5", render_mode="rgb_array", frameskip=4):
    """
    初始化Atari游戏环境，并应用所有必要的包装器。
    
    Args:
        game_name: Atari游戏名称，默认为"ALE/Breakout-v5"
        render_mode: 渲染模式，默认为"rgb_array"
        frameskip: 跳帧数，默认为4
        
    Returns:
        处理过的游戏环境
    """
    try:
        # 创建基础环境
        env = gym.make(
            game_name,
            render_mode=render_mode,
            frameskip=frameskip
        )
        print(f"Environment '{game_name}' loaded successfully!")
        
        # 应用包装器
        env = AutoFireResetEnv(env)
        env = PreprocessAtari(env)
        env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
        
        return env
    except Exception as e:
        print(f"Failed to load environment: {e}")
        sys.exit(1)


def make_evaluation_env(game_name="ALE/Breakout-v5", render_mode="rgb_array", frameskip=4):
    """
    创建用于评估的环境实例。
    
    与训练环境相同，但这是一个单独的实例，用于避免训练和评估之间的干扰。
    
    Args:
        game_name: Atari游戏名称，默认为"ALE/Breakout-v5"
        render_mode: 渲染模式，默认为"rgb_array"
        frameskip: 跳帧数，默认为4
        
    Returns:
        用于评估的环境实例
    """
    return initialize_environment(game_name, render_mode, frameskip)


def make_recording_env(video_folder, game_name="ALE/Breakout-v5", frameskip=4, episode_trigger=lambda e: True):
    """
    创建一个用于录制视频的环境。
    
    Args:
        video_folder: 视频保存的文件夹
        game_name: Atari游戏名称，默认为"ALE/Breakout-v5"
        frameskip: 跳帧数，默认为4
        episode_trigger: 指定何时录制视频的函数，默认为总是录制
        
    Returns:
        能够录制视频的游戏环境
    """
    # 创建基础环境
    base_env = gym.make(game_name, render_mode="rgb_array")
    
    # 添加视频录制包装器
    video_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=video_folder,
        episode_trigger=episode_trigger
    )
    
    # 添加其他必要的包装器
    video_env = AutoFireResetEnv(video_env)
    video_env = PreprocessAtari(video_env)
    video_env = FrameBuffer(video_env, n_frames=4, dim_order='pytorch')
    
    return video_env


# 测试环境创建函数
if __name__ == "__main__":
    env = initialize_environment()
    print(f"Environment created with observation space: {env.observation_space.shape}")
    print(f"Number of actions: {env.action_space.n}")
    
    # 测试一个episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 100:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"Completed {step_count} steps with total reward: {total_reward}")
    env.close()