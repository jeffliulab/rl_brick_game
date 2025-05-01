#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) to play Breakout (ALE/Breakout-v5).

All printed messages and figure texts are in English.
All code comments are in Chinese.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import gc  # 导入垃圾回收模块
import psutil  # 用于内存监控
import tracemalloc  # 用于内存泄漏检测

# ======== Gym & ALE ========
import gymnasium as gym
import ale_py

# ======== PyTorch ========
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======== Other utils ========
import cv2
from tqdm import tqdm, trange
from collections import deque
from pandas import DataFrame
import logging  # 添加日志模块

# 导入绘图和保存相关函数
from plot_utils import (
    save_checkpoint, save_model, save_training_plots, 
    save_training_history, moving_average
)

TOTAL_ITERATION_TIMES = 1000000  # 总迭代次数，先测试100万次，如果能比20万次提升效果，就继续测试500万次

# ========== Part 1: Global configurations, random seeds, device setup ==========

# 设置随机种子，保证实验可复现。并选择CPU或GPU作为计算设备。
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 设置日志
def setup_logger(log_file='output/dqn_training.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('dqn_training')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 限制内存使用
def limit_memory_usage(max_memory_gb=28):
    """限制程序最大内存使用"""
    try:
        import resource
        # 将软限制设为最大内存的90%
        soft_limit = int(max_memory_gb * 0.9 * 1024 * 1024 * 1024)
        # 将硬限制设为最大内存
        hard_limit = int(max_memory_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
        print(f"Memory usage limited to {max_memory_gb}GB")
    except ImportError:
        print("resource module not available, cannot limit memory usage")
    except Exception as e:
        print(f"Failed to set memory limit: {e}")

# 监控内存使用的函数
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

# 打印当前内存使用情况
def print_memory_usage(prefix="", logger=None):
    memory_usage = get_memory_usage()
    message = f"{prefix} Memory usage: {memory_usage:.2f} MB"
    if logger:
        logger.info(message)
    else:
        print(message)
    return memory_usage

# 强制进行垃圾回收
def force_gc(logger=None):
    if logger:
        logger.debug("Performing garbage collection...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 内存泄漏检测
def setup_memory_leak_detection(interval=50000):
    """初始化内存泄漏检测，每隔interval次迭代检查一次"""
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    def check_for_leaks(iteration, logger=None):
        if iteration % interval == 0 and iteration > 0:
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            message = "[ Top 5 memory usage differences ]"
            if logger:
                logger.info(message)
            else:
                print(message)
            
            for stat in top_stats[:5]:
                if logger:
                    logger.info(str(stat))
                else:
                    print(stat)
            return top_stats
        return None
    
    return check_for_leaks

# ========== Part 2: Environment initialization ==========

def initialize_environment(render_mode="rgb_array", frameskip=4):
    """
    Initialize an Atari Breakout environment with given frameskip.
    Prints in English. If environment fails to load, exit.
    """
    try:
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode=render_mode,
            frameskip=frameskip
        )
        print("Environment loaded successfully!")
        return env
    except Exception as e:
        print(f"Failed to load environment: {e}")
        sys.exit(1)

# 在Breakout里，我们常常需要在开局进行一次FIRE动作，以避免训练期间学会"开局发球"。
class AutoFireResetEnv(gym.Wrapper):
    """
    After reset(), automatically fire once.
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, done, truncated, info = self.env.step(1)  # action=1 => FIRE
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

# ========== Part 3: Preprocessing wrappers ==========

class PreprocessAtari(gym.ObservationWrapper):
    """
    Convert color Atari image to grayscale [84x84], normalized to [0,1].
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
    Maintain a buffer of n_frames, stacking them along channel dimension.
    dim_order='pytorch' => (C, H, W).
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

# ========== Part 4: DQN Network, Replay Buffer, Agent ==========

class DQN(nn.Module):
    """
    A Convolutional Neural Network for feature extraction from stacked frames.
    Output: Q(s, a) for each action.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # ========== 卷积层结构 ==========
        # 通常参考DeepMind的Nature DQN配置:
        # Conv2D(32, 8x8, stride=4) => Conv2D(64, 4x4, stride=2) => Conv2D(64, 3x3, stride=1)
        # 之后展平再接全连接层512维, 最后输出n_actions维度
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 计算卷积输出大小
        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # 这里通过构造一个shape一致的全零张量，前向传播一次来获得输出维度
        test_inp = torch.zeros(1, *shape)
        out = self.conv(test_inp)
        return int(np.prod(out.size()))

    def forward(self, x):
        # 前向传播: 卷积 -> 展平 -> 全连接
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """
    A memory-efficient replay buffer storing (s, a, r, s_next, done).
    Uses uint8 for storing frames to save memory.
    """
    def __init__(self, capacity, logger=None):
        self.buffer = deque(maxlen=capacity)
        self.memory_warning_threshold = 0.8  # 内存使用率阈值降低到80%
        self.gc_counter = 0  # 添加计数器来减少垃圾回收频率
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        self.logger = logger
        
        if logger:
            logger.info(f"Total system memory: {self.total_memory:.2f} GB")
        else:
            print(f"Total system memory: {self.total_memory:.2f} GB")
            
        # 记录最后一次紧急内存处理的时间
        self.last_emergency_cleanup = 0

    def __len__(self):
        return len(self.buffer)

    def _check_memory(self):
        """检查内存使用情况，使用更激进的内存管理策略"""
        # 增加计数器，只有每10次调用才真正检查内存
        self.gc_counter += 1
        if self.gc_counter % 10 != 0:
            return False
            
        memory = psutil.virtual_memory()
        memory_used_ratio = memory.percent / 100.0
        
        # 多级内存管理策略
        if memory_used_ratio > 0.95:  # 内存使用率超过95%，紧急情况
            # 确保紧急清理操作不会太频繁（至少间隔1000次添加操作）
            current_time = len(self.buffer)
            if current_time - self.last_emergency_cleanup > 1000:
                self.last_emergency_cleanup = current_time
                message = f"EMERGENCY: Critical memory usage ({memory_used_ratio*100:.1f}%), removing old experiences"
                if self.logger:
                    self.logger.warning(message)
                else:
                    print(message)
                
                # 计算要移除的经验数量（移除20%的缓冲区）
                if len(self.buffer) > 10000:  # 确保有足够的样本
                    remove_count = len(self.buffer) // 5
                    for _ in range(remove_count):
                        self.buffer.popleft()  # 移除最旧的经验
                    
                    message = f"Removed {remove_count} old experiences, buffer size now: {len(self.buffer)}"
                    if self.logger:
                        self.logger.warning(message)
                    else:
                        print(message)
                
                # 强制进行多次垃圾回收
                message = "Performing aggressive garbage collection..."
                if self.logger:
                    self.logger.info(message)
                else:
                    print(message)
                    
                for _ in range(3):  # 连续进行3次垃圾回收
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return True
                
        elif memory_used_ratio > 0.85:  # 内存使用率超过85%，常规垃圾回收
            message = f"Warning: High memory usage ({memory_used_ratio*100:.1f}%), triggering garbage collection"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
                
            force_gc(self.logger)  # 正常垃圾回收
            return True
            
        return False

    def add(self, s, a, r, s_next, done):
        """
        添加经验到缓冲区，将浮点状态转为uint8节省内存
        """
        # 检查并转换状态为uint8以节省内存
        if isinstance(s, np.ndarray) and s.dtype == np.float32:
            s = (s * 255).astype(np.uint8)
        if isinstance(s_next, np.ndarray) and s_next.dtype == np.float32:
            s_next = (s_next * 255).astype(np.uint8)
            
        self.buffer.append((s, a, r, s_next, done))
        
        # 减少内存检查频率，只在添加较多样本后才检查
        if len(self.buffer) % 1000 == 0:  # 从10000改为1000
            self._check_memory()

    def sample(self, batch_size, max_attempts=3):
        """
        带有内存安全机制的采样函数，会在内存不足时自动减小批量大小
        """
        if len(self.buffer) < batch_size:
            return None
            
        current_batch_size = batch_size
        attempts = 0
        
        while attempts < max_attempts:
            try:
                batch = random.sample(self.buffer, current_batch_size)
                
                # 批量处理转换，分步进行以减少峰值内存
                # 1. 先处理状态
                s_batch = []
                for s, _, _, _, _ in batch:
                    s_batch.append(s)
                s = np.array(s_batch, dtype=np.uint8).astype(np.float32) / 255.0
                s_t = torch.from_numpy(s).to(device)
                del s_batch  # 释放中间变量
                
                # 2. 处理动作
                a = np.array([item[1] for item in batch], dtype=np.int64)
                a_t = torch.from_numpy(a).to(device)
                
                # 3. 处理奖励
                r = np.array([item[2] for item in batch], dtype=np.float32)
                r_t = torch.from_numpy(r).to(device)
                
                # 4. 处理下一状态
                s_next_batch = []
                for _, _, _, s_next, _ in batch:
                    s_next_batch.append(s_next)
                s_next = np.array(s_next_batch, dtype=np.uint8).astype(np.float32) / 255.0
                s_next_t = torch.from_numpy(s_next).to(device)
                del s_next_batch  # 释放中间变量
                
                # 5. 处理结束标志
                done = np.array([item[4] for item in batch], dtype=np.float32)
                done_t = torch.from_numpy(done).to(device)
                
                return s_t, a_t, r_t, s_next_t, done_t
                
            except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
                # 内存不足，减小批量大小并重试
                attempts += 1
                force_gc(self.logger)  # 强制垃圾回收
                current_batch_size = current_batch_size // 2
                
                message = f"Memory error: {e}, reducing batch size to {current_batch_size}"
                if self.logger:
                    self.logger.warning(message)
                else:
                    print(message)
                
        # 如果多次尝试后仍然失败，返回一个非常小的批次
        message = "Multiple sampling attempts failed, using minimal batch size"
        if self.logger:
            self.logger.warning(message)
        else:
            print(message)
            
        return self.sample(8, max_attempts=1)  # 尝试用极小的批量大小


class DQNAgent:
    """
    DQN agent with a policy network and a target network, plus epsilon-greedy.
    """
    def __init__(self, state_shape, n_actions, epsilon=1.0):
        self.epsilon = epsilon
        self.n_actions = n_actions

        # 策略网络 (policy_net)
        self.policy_net = DQN(state_shape, n_actions).to(device)
        # 目标网络 (target_net)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.update_target_network()

        # target_net参数冻结，不参加梯度计算
        for p in self.target_net.parameters():
            p.requires_grad = False

    def update_target_network(self):
        # 复制policy_net的参数到target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_qvalues(self, states):
        with torch.no_grad():
            inp = torch.from_numpy(states).float().to(device)
            qvals = self.policy_net(inp)
            return qvals.cpu().numpy()

    def sample_actions(self, qvalues):
        # \epsilon-greedy: 以epsilon概率随机动作，否则选Q值最大的动作
        batch_size = qvalues.shape[0]
        random_actions = np.random.choice(self.n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=1)
        explore = np.random.rand(batch_size) < self.epsilon
        return np.where(explore, random_actions, best_actions)


# ========== Part 5: Training & Evaluation functions ==========

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, logger=None):
    """
    Let the agent play 'n_games' in the environment, return average total reward.
    If 'greedy=True', use argmax(Q); else use epsilon-greedy.
    """
    rewards = []
    for _ in range(n_games):
        obs, info = env.reset()
        ep_reward = 0
        for _ in range(t_max):
            state_batch = np.array([obs], dtype=np.float32)
            qvals = agent.get_qvalues(state_batch)
            if greedy:
                action = qvals.argmax(axis=1)[0]
            else:
                action = agent.sample_actions(qvals)[0]
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    
    avg_reward = np.mean(rewards)
    if logger:
        logger.info(f"Evaluation: average reward over {n_games} games: {avg_reward:.2f}")
    return avg_reward


def play_and_record(agent, env, replay_buffer, n_steps=1, logger=None):
    """
    Interact with the environment for n_steps. Store transitions into the replay buffer.
    Returns the sum of rewards over these n_steps.
    """
    s = env.framebuffer
    reward_sum = 0.0

    for _ in range(n_steps):
        state_batch = np.array([s], dtype=np.float32)
        qvals = agent.get_qvalues(state_batch)
        action = agent.sample_actions(qvals)[0]

        s_next, reward, done, info = env.step(action)
        replay_buffer.add(s, action, reward, s_next, float(done))

        reward_sum += reward

        if done:
            s_reset, _ = env.reset()
            s = s_reset
        else:
            s = s_next

    return reward_sum


def train_on_batch(agent, optimizer, replay_buffer, batch_size=64, gamma=0.99, logger=None):
    """
    Sample a batch from replay buffer and do one gradient step.
    Now includes error handling for memory issues.
    """
    if len(replay_buffer) < batch_size:
        if logger:
            logger.debug("Replay buffer too small for sampling")
        return 0.0

    # 获取一个批量的数据，如果内存不足会自动减小批量大小
    batch_data = replay_buffer.sample(batch_size)
    
    # 如果采样失败，返回0损失
    if batch_data is None:
        message = "Batch sampling returned None, skipping training step"
        if logger:
            logger.warning(message)
        else:
            print(message)
        return 0.0
        
    states, actions, rewards, next_states, dones = batch_data

    try:
        # Q_policy(s,a)
        qvals = agent.policy_net(states)
        current_q = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算TD目标 y = r + gamma * max_{a'} Q_target(s', a') * (1 - done)
        with torch.no_grad():
            qvals_next = agent.target_net(next_states)
            max_q_next = qvals_next.max(dim=1)[0]
            target_q = rewards + gamma * max_q_next * (1 - dones)

        # 使用smooth_l1_loss(Huber Loss)来衡量 current_q 与 target_q 的差异
        loss = F.smooth_l1_loss(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，避免梯度爆炸
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()
        return loss.item()
        
    except RuntimeError as e:
        message = f"Training error: {e}"
        if logger:
            logger.error(message)
        else:
            print(message)
        force_gc(logger)  # 强制垃圾回收
        return 0.0


# ========== Part 6: Main training script ==========

def main():
    # 限制最大内存使用
    limit_memory_usage(28)
    
    # 设置日志
    logger = setup_logger()
    logger.info("Starting DQN training script")
    
    # 启动内存泄漏检测
    leak_detector = setup_memory_leak_detection(interval=50000)
    
    # 1) 初始化环境
    raw_env = initialize_environment(render_mode="rgb_array", frameskip=4)
    raw_env = AutoFireResetEnv(raw_env)
    env = PreprocessAtari(raw_env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')

    # 2) 准备DQNAgent, 经验回放等
    n_actions = env.action_space.n
    logger.info(f"Number of actions: {n_actions}")
    state_shape = env.observation_space.shape

    # 打印初始内存使用情况
    print_memory_usage("Initial", logger)

    # 更合理的回放缓冲区大小，防止内存溢出
    replay_buffer_size = 100000  # 从500,000减小到100,000
    
    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    # 添加学习率调度器 - 每100万步学习率减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size, logger=logger)

    obs, info = env.reset()

    # 先预填充一些回放缓冲区, 让网络能sample到足够多样的经验
    logger.info("Filling replay buffer with 50000 steps, please wait...")
    pbar = tqdm(total=50000, desc="Buffer Filling")
    steps_done = 0
    while steps_done < 50000:
        steps_to_do = min(1000, 50000 - steps_done)
        reward = play_and_record(agent, env, replay_buffer, n_steps=steps_to_do, logger=logger)
        steps_done += steps_to_do
        pbar.update(steps_to_do)
        
        # 每10000步检查一次内存
        if steps_done % 10000 == 0:
            print_memory_usage(f"Buffer fill {steps_done} steps", logger)
            force_gc(logger)  # 强制垃圾回收
            
    pbar.close()

    # 下面这些列表用于记录训练/评估指标, 以便可视化
    mean_rw_history = []       # 评估时的平均reward
    loss_history = []          # 训练过程中的TD损失
    train_reward_history = []  # 每次迭代play_and_record返回的奖励
    epsilon_history = []       # epsilon随迭代变化的曲线
    memory_usage_history = []  # 内存使用历史

    # 超参数设定
    gamma = 0.99
    batch_size = 16  # 从32减小到16
    target_update_freq = 10000
    
    # 线性衰减的epsilon参数
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay_steps = 1000000  # 前100万步线性衰减

    eval_freq = 50000        # 每多少迭代做一次评估
    memory_check_freq = 10000  # 每多少迭代检查一次内存
    num_iterations = TOTAL_ITERATION_TIMES # 总迭代次数

    # 添加自动保存点，用于训练中断后恢复
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 修改保存频率设置
    early_save_points = [50, 100]  # 添加早期保存点，用于测试保存机制
    regular_save_freq = 50000      # 将常规保存频率改为50000

    logger.info("Start DQN training...")
    for i in trange(num_iterations):
        # 每次迭代先与环境交互10步
        train_reward = play_and_record(agent, env, replay_buffer, n_steps=10, logger=logger)
        train_reward_history.append(train_reward)

# 然后sample一个batch做一次训练
        loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma, logger=logger)
        loss_history.append(loss)

        # 线性衰减epsilon
        if i < epsilon_decay_steps:
            agent.epsilon = epsilon_start - (epsilon_start - epsilon_final) * (i / epsilon_decay_steps)
        else:
            agent.epsilon = epsilon_final

        # 隔一定迭代频率更新target_net
        if i % target_update_freq == 0:
            agent.update_target_network()
            logger.info(f"Iteration {i}: Target network updated")
            
        # 每1000步调整一次学习率
        if i % 1000 == 0:
            scheduler.step()

        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)
        
        # 定期监控内存使用
        if i % memory_check_freq == 0:
            memory_usage = print_memory_usage(f"Iteration {i}", logger)
            memory_usage_history.append(memory_usage)
            logger.info(f"Iteration {i}, Buffer size: {len(replay_buffer)}, Epsilon: {agent.epsilon:.3f}")
            
            # 检查内存泄漏
            leak_detector(i, logger)
            
            # 强制垃圾回收
            force_gc(logger)
            
            # 如果内存使用量过高，主动清理一部分缓冲区
            memory = psutil.virtual_memory()
            if memory.percent > 90:  # 整体内存使用超过90%
                # 移除10%的缓冲区
                if len(replay_buffer.buffer) > 10000:  # 确保有足够的样本
                    to_remove = len(replay_buffer.buffer) // 10
                    for _ in range(to_remove):
                        replay_buffer.buffer.popleft()
                    logger.warning(f"HIGH MEMORY ALERT: Removed {to_remove} samples from buffer")
                    force_gc(logger)  # 再次强制垃圾回收

        # 定期评估
        if i % eval_freq == 0:
            # 先进行垃圾回收以确保评估有足够内存
            force_gc(logger)
            
            try:
                eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
                eval_env = AutoFireResetEnv(eval_env)
                eval_env = PreprocessAtari(eval_env)
                eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')

                mean_reward = evaluate(eval_env, agent, n_games=3, greedy=True, logger=logger)
                mean_rw_history.append(mean_reward)

                logger.info(f"Iteration {i}, Eval average reward: {mean_reward:.2f}, "
                          f"Replay size: {len(replay_buffer)}, "
                          f"Epsilon: {agent.epsilon:.3f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")
                          
                # 关闭评估环境以释放资源
                eval_env.close()
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                
            # 评估后再次进行垃圾回收
            force_gc(logger)
        
        # 检查是否是早期保存点或常规保存点
        should_save = (i in early_save_points) or (i > 0 and i % regular_save_freq == 0)
        
        if should_save:
            # 强制垃圾回收以确保有足够内存
            force_gc(logger)
            
            # 调用绘图和保存函数
            try:
                save_checkpoint(i, agent, optimizer, scheduler, 
                              mean_rw_history, loss_history, train_reward_history, 
                              epsilon_history, memory_usage_history, 
                              checkpoint_dir, logger)
                              
                save_model(i, agent, logger)
                
                save_training_plots(i, mean_rw_history, train_reward_history, loss_history, 
                                  epsilon_history, memory_usage_history, memory_check_freq, 
                                  logger)
                                  
                save_training_history(i, mean_rw_history, train_reward_history, loss_history, 
                                    epsilon_history, memory_usage_history, logger)
            except Exception as e:
                logger.error(f"Failed to save at iteration {i}: {e}")

    # ========== 训练结束, 保存最终模型 ==========
    force_gc(logger)  # 确保有足够内存保存模型
    
    try:
        # 保存最终模型
        save_model("final", agent, logger)
        
        # 保存最终训练曲线
        save_training_plots("final", mean_rw_history, train_reward_history, loss_history, 
                           epsilon_history, memory_usage_history, memory_check_freq, logger)
                           
        # 保存最终训练历史数据
        save_training_history("final", mean_rw_history, train_reward_history, loss_history, 
                             epsilon_history, memory_usage_history, logger)
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

    # ========== 录制一段评估视频 ==========
    try:
        agent.epsilon = 0  # 测试时设为贪心
        os.makedirs("output/videos", exist_ok=True)

        video_env = gym.wrappers.RecordVideo(
            initialize_environment(render_mode="rgb_array"),
            video_folder="output/videos",
            episode_trigger=lambda e: True
        )
        video_env = AutoFireResetEnv(video_env)
        video_env = PreprocessAtari(video_env)
        video_env = FrameBuffer(video_env, n_frames=4, dim_order='pytorch')

        final_rewards = [evaluate(video_env, agent, n_games=1, greedy=True, logger=logger)]
        video_env.close()
        logger.info(f"Final reward (one episode): {np.mean(final_rewards):.2f}")
    except Exception as e:
        logger.error(f"Video recording failed: {e}")


# 用于从检查点恢复训练的函数
def resume_from_checkpoint(checkpoint_path, logger=None):
    """从检查点文件恢复训练状态"""
    if logger:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # 1) 初始化环境
    raw_env = initialize_environment(render_mode="rgb_array", frameskip=4)
    raw_env = AutoFireResetEnv(raw_env)
    env = PreprocessAtari(raw_env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    
    # 2) 加载检查点
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    
    checkpoint = torch.load(checkpoint_path)
    
    # 3) 初始化Agent和优化器
    agent = DQNAgent(state_shape, n_actions, epsilon=checkpoint['epsilon'])
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.5)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 4) 重新创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100000, logger=logger)  # 更小的容量
    
    # 5) 重新填充回放缓冲区
    if logger:
        logger.info("Refilling replay buffer...")
    else:
        print("Refilling replay buffer...")
        
    pbar = tqdm(total=50000, desc="Buffer Refilling")
    steps_done = 0
    while steps_done < 50000:
        steps_to_do = min(1000, 50000 - steps_done)
        reward = play_and_record(agent, env, replay_buffer, n_steps=steps_to_do, logger=logger)
        steps_done += steps_to_do
        pbar.update(steps_to_do)
    pbar.close()
    
    # 6) 恢复保存的历史数据
    mean_rw_history = checkpoint.get('mean_rw_history', [])
    loss_history = checkpoint.get('loss_history', [])
    train_reward_history = checkpoint.get('train_reward_history', [])
    epsilon_history = checkpoint.get('epsilon_history', [])
    memory_usage_history = checkpoint.get('memory_usage_history', [])
    
    # 7) 返回恢复的状态
    return (env, agent, optimizer, scheduler, replay_buffer, 
            checkpoint['iteration'], mean_rw_history, loss_history, 
            train_reward_history, epsilon_history, memory_usage_history)

# 执行真正的从检查点恢复训练的函数
def continue_training_from_checkpoint(checkpoint_path, num_iterations=None):
    """实际执行从检查点恢复训练的函数"""
    # 设置日志
    logger = setup_logger(log_file='output/dqn_resumed_training.log')
    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # 启动内存泄漏检测
    leak_detector = setup_memory_leak_detection(interval=50000)
    
    # 从检查点恢复状态
    try:
        (env, agent, optimizer, scheduler, replay_buffer, 
         start_iteration, mean_rw_history, loss_history, 
         train_reward_history, epsilon_history, memory_usage_history) = resume_from_checkpoint(checkpoint_path, logger)
        
        logger.info(f"Successfully restored training state from iteration {start_iteration}")
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        logger.info("Starting fresh training...")
        main()
        return
    
    # 设置迭代次数
    if num_iterations is None:
        num_iterations = TOTAL_ITERATION_TIMES
    
    # 超参数设定
    gamma = 0.99
    batch_size = 16  # 从32减小到16，减轻内存负担
    target_update_freq = 10000
    
    # 线性衰减的epsilon参数 - 从当前epsilon继续衰减
    epsilon_final = 0.01
    epsilon_decay_steps = 1000000  # 前100万步线性衰减
    
    eval_freq = 50000        # 每多少迭代做一次评估
    memory_check_freq = 10000  # 每多少迭代检查一次内存
    
    # 添加自动保存点，用于训练中断后恢复
    checkpoint_dir = "output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 修改保存频率设置
    regular_save_freq = 50000  # 每50000次迭代保存一次
    
    logger.info(f"Continuing DQN training from iteration {start_iteration}...")
    for i in trange(start_iteration, num_iterations):
        # 每次迭代先与环境交互10步
        train_reward = play_and_record(agent, env, replay_buffer, n_steps=10, logger=logger)
        train_reward_history.append(train_reward)
        
        # 然后sample一个batch做一次训练
        loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma, logger=logger)
        loss_history.append(loss)
        
        # 线性衰减epsilon
        if i < epsilon_decay_steps:
            agent.epsilon = max(agent.epsilon - (1.0 - epsilon_final) / epsilon_decay_steps, epsilon_final)
        else:
            agent.epsilon = epsilon_final
            
        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)
        
        # 隔一定迭代频率更新target_net
        if i % target_update_freq == 0:
            agent.update_target_network()
            logger.info(f"Iteration {i}: Target network updated")
            
        # 每1000步调整一次学习率
        if i % 1000 == 0:
            scheduler.step()
            
        # 定期监控内存使用
        if i % memory_check_freq == 0:
            memory_usage = print_memory_usage(f"Iteration {i}", logger)
            memory_usage_history.append(memory_usage)
            logger.info(f"Iteration {i}, Buffer size: {len(replay_buffer)}, Epsilon: {agent.epsilon:.3f}")
            
            # 检查内存泄漏
            leak_detector(i, logger)
            
            # 强制垃圾回收
            force_gc(logger)
            
            # 如果内存使用量过高，主动清理一部分缓冲区
            memory = psutil.virtual_memory()
            if memory.percent > 90:  # 整体内存使用超过90%
                # 移除10%的缓冲区
                if len(replay_buffer.buffer) > 10000:  # 确保有足够的样本
                    to_remove = len(replay_buffer.buffer) // 10
                    for _ in range(to_remove):
                        replay_buffer.buffer.popleft()
                    logger.warning(f"HIGH MEMORY ALERT: Removed {to_remove} samples from buffer")
                    force_gc(logger)  # 再次强制垃圾回收
        
        # 定期评估
        if i % eval_freq == 0:
            # 先进行垃圾回收以确保评估有足够内存
            force_gc(logger)
            
            try:
                eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
                eval_env = AutoFireResetEnv(eval_env)
                eval_env = PreprocessAtari(eval_env)
                eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')

                mean_reward = evaluate(eval_env, agent, n_games=3, greedy=True, logger=logger)
                mean_rw_history.append(mean_reward)

                logger.info(f"Iteration {i}, Eval average reward: {mean_reward:.2f}, "
                            f"Replay size: {len(replay_buffer)}, "
                            f"Epsilon: {agent.epsilon:.3f}, "
                            f"LR: {scheduler.get_last_lr()[0]:.6f}")
                
                # 关闭评估环境以释放资源
                eval_env.close()
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                
            # 评估后再次进行垃圾回收
            force_gc(logger)
            
        # 检查是否应该保存
        if i > start_iteration and i % regular_save_freq == 0:
            # 强制垃圾回收以确保有足够内存
            force_gc(logger)
            
            # 调用绘图和保存函数
            try:
                save_checkpoint(i, agent, optimizer, scheduler, 
                              mean_rw_history, loss_history, train_reward_history, 
                              epsilon_history, memory_usage_history, 
                              checkpoint_dir, logger)
                              
                save_model(i, agent, logger)
                
                save_training_plots(i, mean_rw_history, train_reward_history, loss_history, 
                                  epsilon_history, memory_usage_history, memory_check_freq, 
                                  logger)
                                  
                save_training_history(i, mean_rw_history, train_reward_history, loss_history, 
                                    epsilon_history, memory_usage_history, logger)
            except Exception as e:
                logger.error(f"Failed to save at iteration {i}: {e}")
                
    # 保存最终结果
    force_gc(logger)
    try:
        # 保存最终模型
        save_model("final_resumed", agent, logger)
        
        # 保存最终训练曲线
        save_training_plots("final_resumed", mean_rw_history, train_reward_history, loss_history, 
                           epsilon_history, memory_usage_history, memory_check_freq, logger)
                           
        # 保存最终训练历史数据
        save_training_history("final_resumed", mean_rw_history, train_reward_history, loss_history, 
                             epsilon_history, memory_usage_history, logger)
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

# 处理断点续训的入口函数
def main_with_recovery():
    """主入口函数，支持从检查点恢复训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN Breakout Training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--iterations', type=int, help='Number of iterations to train for', default=TOTAL_ITERATION_TIMES)
    args = parser.parse_args()
    
    if args.resume and os.path.exists(args.resume):
        try:
            continue_training_from_checkpoint(args.resume, args.iterations)
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")
            print("Starting fresh training...")
            main()
    else:
        main()

if __name__ == "__main__":
    try:
        main_with_recovery()
    except Exception as e:
        print(f"Fatal error: {e}")
        # 打印完整的异常堆栈
        import traceback
        traceback.print_exc()