#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) to play Breakout (ALE/Breakout-v5).

All printed messages and figure texts are in English.
All code comments are in Chinese.
"""

import os
# 设置OpenMP环境变量，解决多重初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 限制OpenMP线程数，减少潜在冲突
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
from tqdm import trange
from collections import deque
from pandas import DataFrame
import traceback

# 检测是否可以录制视频
ENABLE_VIDEO_RECORDING = False
try:
    import moviepy
    ENABLE_VIDEO_RECORDING = True
except ImportError:
    print("MoviePy not installed. Video recording will be disabled.")
    print("To enable video recording, run: pip install 'gymnasium[other]'")

# ========== Part 1: Global configurations, random seeds, device setup ==========

# 以下注释均为中文，为了方便理解：
# 设置随机种子，保证实验可复现。并选择CPU或GPU作为计算设备。
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    # 启用 cudnn 自动优化
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 用于绘制曲线时的指数加权平均函数 (moving average)
def moving_average(x, span, **kw):
    return (
        DataFrame({"x": np.asarray(x)})
        .x.ewm(span=span, **kw)
        .mean()
        .values
    )

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
    A simple replay buffer storing (s, a, r, s_next, done).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        """批量采样并一次性转移到GPU，减少CPU-GPU通信开销"""
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            s, a, r, next_s, done = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(next_s)
            dones.append(done)
            
        # 优化内存使用: 分批转换和转移到GPU, 每次处理小批量
        def to_tensor(data, dtype=np.float32):
            data_np = np.array(data, dtype=dtype)
            return torch.from_numpy(data_np).to(device)
            
        # 分别转换和转移，减少内存使用
        s_t = to_tensor(states)
        a_t = to_tensor(actions, dtype=np.int64)
        r_t = to_tensor(rewards)
        s_next_t = to_tensor(next_states)
        done_t = to_tensor(dones)
        
        # 清理临时列表以释放内存
        del states, actions, rewards, next_states, dones
        
        return s_t, a_t, r_t, s_next_t, done_t

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

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
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
    return np.mean(rewards)


def play_and_record(agent, env, replay_buffer, n_steps=1):
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

def train_on_batch(agent, optimizer, replay_buffer, batch_size=64, gamma=0.99):
    """
    Sample a batch from replay buffer and do one gradient step.

    这里我们采用的是时序差分 (Temporal Difference) 学习:
    在Q-learning的更新公式中, 目标值 y 可写为:
        y = r + gamma * max_{a'} Q_target(s', a')
    在DQN中, 我们用 target_net 来计算 max_{a'} Q_target(s', a'), 以减少自举偏差.

    损失函数 (loss) 通常用 HuberLoss (smooth_l1_loss), 表达形式大致是:
        L = Huber( Q_policy(s,a) - y )

    其中 Q_policy(s,a) 是当前策略网络的输出, y 是上述TD目标值.
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    # 使用try块捕获可能的内存或GPU错误
    try:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

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

        # 可选：梯度裁剪，避免梯度爆炸
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()
        return loss.item()
    
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"\nWARNING: GPU out of memory during training. Reducing batch size to {batch_size//2}")
            # 释放一些内存
            torch.cuda.empty_cache()
            # 减小批次大小并重试
            if batch_size > 32:
                return train_on_batch(agent, optimizer, replay_buffer, batch_size//2, gamma)
            else:
                print("Cannot reduce batch size further. Skipping this training step.")
                return 0.0
        else:
            print(f"\nError during training: {e}")
            return 0.0

# 录制游戏视频的函数
def record_game_video(agent, output_path, video_name="gameplay_video"):
    """录制一段游戏视频"""
    if not ENABLE_VIDEO_RECORDING:
        print(f"Video recording disabled (MoviePy not installed).")
        return None
        
    try:
        video_env = gym.wrappers.RecordVideo(
            initialize_environment(render_mode="rgb_array"),
            video_folder=output_path,
            name_prefix=video_name,
            episode_trigger=lambda e: True
        )
        video_env = AutoFireResetEnv(video_env)
        video_env = PreprocessAtari(video_env)
        video_env = FrameBuffer(video_env, n_frames=4, dim_order='pytorch')

        reward = evaluate(video_env, agent, n_games=1, greedy=True)
        video_env.close()
        return reward
    except Exception as e:
        print(f"Error recording video: {e}")
        return None

# 记录超参数和配置到文件
def save_hyperparameters(output_path, params):
    """将超参数保存到文件"""
    with open(os.path.join(output_path, "hyperparameters.txt"), "w") as f:
        f.write("DQN Training Hyperparameters\n")
        f.write("===========================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

# ========== Part 6: Main training script ==========

def main():
    # 创建输出文件夹结构
    output_path = "output2"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "self_check"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "self_check", "models"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "self_check", "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "self_check", "plots"), exist_ok=True)
    print("Output directories created.")

    # 1) 初始化环境
    raw_env = initialize_environment(render_mode="rgb_array", frameskip=4)
    raw_env = AutoFireResetEnv(raw_env)
    env = PreprocessAtari(raw_env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')

    # 2) 准备DQNAgent, 经验回放等
    n_actions = env.action_space.n
    print("Number of actions:", n_actions)
    state_shape = env.observation_space.shape

    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    
    # 增加经验回放缓冲区大小以匹配500万次训练
    replay_buffer = ReplayBuffer(capacity=500_000)

    obs, info = env.reset()

    # 先预填充一些回放缓冲区, 让网络能sample到足够多样的经验
    print("Filling replay buffer, please wait...")
    play_and_record(agent, env, replay_buffer, n_steps=10000)

    # 下面这些列表用于记录训练/评估指标, 以便可视化
    mean_rw_history = []       # 评估时的平均reward
    loss_history = []          # 训练过程中的TD损失
    train_reward_history = []  # 每次迭代play_and_record返回的奖励
    epsilon_history = []       # epsilon随迭代变化的曲线
    
    # 添加回合奖励跟踪
    episode_rewards = []       # 完整回合的奖励
    episode_steps = []         # 每个回合对应的步数
    current_episode_reward = 0 # 当前回合的累计奖励
    episode_count = 0          # 回合计数

    # 超参数设定 - 优化GPU使用
    gamma = 0.99
    batch_size = 128           # 减小批次大小以减少内存使用
    target_update_freq = 500
    training_freq = 4          # 每收集一批数据训练几次
    collection_steps = 20      # 每次收集数据的步数
    
    # 由于训练步数增加到500万，降低ε衰减速率
    epsilon_decay = 0.9999
    min_epsilon = 0.01

    eval_freq = 5000             # 每多少步做一次评估
    log_freq = 100               # 每多少步记录一次详细日志
    save_freq = 50000            # 每多少步保存一次模型
    total_steps_target = 5_000_000  # 总训练步数目标
    num_iterations = total_steps_target // collection_steps 
    
    # 自检阶段参数
    selfcheck_freq = 10          # 自检阶段每多少次迭代进行一次全面检查
    selfcheck_iterations = 100   # 自检阶段的迭代次数
    
    # 超参数保存
    hyperparams = {
        "gamma": gamma,
        "batch_size": batch_size,
        "target_update_freq": target_update_freq,
        "training_freq": training_freq,
        "collection_steps": collection_steps,
        "epsilon_decay": epsilon_decay,
        "min_epsilon": min_epsilon,
        "replay_buffer_size": replay_buffer.buffer.maxlen,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "total_steps_target": total_steps_target,
        "video_recording_enabled": ENABLE_VIDEO_RECORDING
    }
    save_hyperparameters(output_path, hyperparams)

    # 详细日志文件准备
    main_log_file = os.path.join(output_path, "training_log.csv")
    with open(main_log_file, 'w') as f:
        f.write("step,iteration,episode,reward,avg_reward_10,avg_reward_100,epsilon,loss,replay_size,eval_reward,train_fps,timestamp\n")
    
    # 自检日志文件准备
    selfcheck_log_file = os.path.join(output_path, "self_check", "selfcheck_log.csv")
    with open(selfcheck_log_file, 'w') as f:
        f.write("step,iteration,episode,reward,epsilon,loss,replay_size,eval_reward,video_reward,model_saved,timestamp\n")

    print(f"Starting DQN training with self-check phase ({selfcheck_iterations} iterations) followed by main training...")
    start_time = time.time()
    total_steps = 0
    is_selfcheck_phase = True
    
    # 主训练循环
    for i in trange(num_iterations):
        # 定期清理GPU内存
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 每次迭代收集多步数据
        iteration_start_time = time.time()
        train_reward = play_and_record(agent, env, replay_buffer, n_steps=collection_steps)
        train_reward_history.append(train_reward)
        total_steps += collection_steps
        
        # 记录每次与环境交互的回合结束情况
        if train_reward > 0 and train_reward > current_episode_reward:
            # 可能是回合结束
            episode_rewards.append(train_reward)
            episode_steps.append(total_steps)
            current_episode_reward = 0
            episode_count += 1
        else:
            current_episode_reward += train_reward

        # 多次训练，充分利用GPU
        avg_loss = 0
        for _ in range(training_freq):
            try:
                loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma)
                avg_loss += loss / training_freq
            except Exception as e:
                print(f"\nError during training: {e}")
                traceback.print_exc()
                # 继续训练而不中断
                continue
            
        loss_history.append(avg_loss)

        # 隔一定迭代频率更新target_net, 并衰减epsilon
        if i % (target_update_freq // collection_steps) == 0:
            agent.update_target_network()
            agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)

        # 记录当前epsilon
        epsilon_history.append(agent.epsilon)
        
        # 计算训练速度
        iteration_time = time.time() - iteration_start_time
        train_fps = collection_steps / iteration_time if iteration_time > 0 else 0
        
        # ===== 自检阶段逻辑 =====
        if is_selfcheck_phase and i < selfcheck_iterations:
            if i % selfcheck_freq == 0:
                try:
                    # 进行评估
                    eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
                    eval_env = AutoFireResetEnv(eval_env)
                    eval_env = PreprocessAtari(eval_env)
                    eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')
                    eval_reward = evaluate(eval_env, agent, n_games=2, greedy=True)
                    mean_rw_history.append(eval_reward)
                    
                    # 录制视频
                    video_path = os.path.join(output_path, "self_check", "videos")
                    video_reward = record_game_video(agent, video_path, f"selfcheck_{total_steps}")
                    
                    # 保存模型
                    model_path = os.path.join(output_path, "self_check", "models", f"dqn_model_{total_steps}.pt")
                    torch.save(agent.policy_net.state_dict(), model_path)
                    
                    # 保存训练曲线
                    plt.figure(figsize=(12, 10))
                    
                    # 回合奖励曲线
                    plt.subplot(2, 2, 1)
                    plt.title(f"Episode Rewards (Self-check {total_steps})")
                    if episode_rewards:
                        plt.plot(episode_steps, episode_rewards, 'b-', alpha=0.3)
                        if len(episode_rewards) >= 10:
                            moving_avg = pd.Series(episode_rewards).rolling(10).mean()
                            plt.plot(episode_steps[:len(moving_avg)], moving_avg, 'r-')
                    plt.grid()
                    
                    # 评估奖励曲线
                    plt.subplot(2, 2, 2)
                    plt.title("Evaluation Rewards")
                    plt.plot(mean_rw_history)
                    plt.grid()
                    
                    # 训练损失曲线
                    plt.subplot(2, 2, 3)
                    plt.title("Training Loss")
                    plt.plot(loss_history)
                    plt.grid()
                    
                    # Epsilon曲线
                    plt.subplot(2, 2, 4)
                    plt.title("Epsilon")
                    plt.plot(epsilon_history)
                    plt.grid()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_path, "self_check", "plots", f"training_curves_{total_steps}.png"))
                    plt.close()
                    
                    # 记录到自检日志
                    with open(selfcheck_log_file, 'a') as f:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"{total_steps},{i},{episode_count},{train_reward},{agent.epsilon:.4f},{avg_loss:.6f},{len(replay_buffer)},{eval_reward:.2f},{video_reward if video_reward else 'N/A'},True,{timestamp}\n")
                    
                    print(f"\nSelf-check {i}/{selfcheck_iterations}: Step {total_steps}, Eval: {eval_reward:.2f}, Video: {video_reward if video_reward else 'N/A'}, Episodes: {episode_count}")
                
                except Exception as e:
                    print(f"\nError during self-check: {e}")
                    with open(selfcheck_log_file, 'a') as f:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"{total_steps},{i},{episode_count},{train_reward},{agent.epsilon:.4f},{avg_loss:.6f},{len(replay_buffer)},ERROR,ERROR,False,{timestamp}\n")
            
            # 如果完成了自检阶段，打印总结并转入正式训练
            if i == selfcheck_iterations - 1:
                is_selfcheck_phase = False
                print("\n" + "="*80)
                print(f"Self-check phase completed. {selfcheck_iterations} iterations ({total_steps} steps) completed.")
                print(f"Agent collected {episode_count} episodes with average reward: {np.mean(episode_rewards) if episode_rewards else 'N/A'}")
                print(f"Current epsilon: {agent.epsilon:.4f}, Replay buffer size: {len(replay_buffer)}")
                print(f"Proceeding to main training phase...")
                print("="*80 + "\n")
        
        # ===== 正式训练阶段逻辑 =====
        else:
            # 每log_freq步记录详细训练日志
            if total_steps % log_freq < collection_steps and total_steps > 0:
                # 计算平均奖励
                recent_rewards_10 = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                recent_rewards_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                avg_reward_10 = sum(recent_rewards_10) / len(recent_rewards_10) if recent_rewards_10 else 0
                avg_reward_100 = sum(recent_rewards_100) / len(recent_rewards_100) if recent_rewards_100 else 0
                
                # 记录到训练日志
                with open(main_log_file, 'a') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    recent_reward = episode_rewards[-1] if episode_rewards else 0
                    # 只在评估步骤才进行评估，否则用-1表示没有评估
                    eval_reward = -1
                    f.write(f"{total_steps},{i},{episode_count},{recent_reward:.2f},{avg_reward_10:.2f},{avg_reward_100:.2f},{agent.epsilon:.4f},{avg_loss:.6f},{len(replay_buffer)},{eval_reward:.2f},{train_fps:.1f},{timestamp}\n")
            
            # 定期评估
            if total_steps % eval_freq < collection_steps and total_steps > 0:
                try:
                    # 强制GPU同步，确保所有操作完成
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
                    eval_env = AutoFireResetEnv(eval_env)
                    eval_env = PreprocessAtari(eval_env)
                    eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')

                    eval_reward = evaluate(eval_env, agent, n_games=3, greedy=True)
                    mean_rw_history.append(eval_reward)
                    
                    # 记录评估结果到训练日志
                    with open(main_log_file, 'a') as f:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        recent_reward = episode_rewards[-1] if episode_rewards else 0
                        recent_rewards_10 = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                        recent_rewards_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                        avg_reward_10 = sum(recent_rewards_10) / len(recent_rewards_10) if recent_rewards_10 else 0
                        avg_reward_100 = sum(recent_rewards_100) / len(recent_rewards_100) if recent_rewards_100 else 0
                        f.write(f"{total_steps},{i},{episode_count},{recent_reward:.2f},{avg_reward_10:.2f},{avg_reward_100:.2f},{agent.epsilon:.4f},{avg_loss:.6f},{len(replay_buffer)},{eval_reward:.2f},{train_fps:.1f},{timestamp}\n")

                    print(f"\nStep {total_steps}, Eval average reward: {eval_reward:.2f}, "
                          f"Avg10: {avg_reward_10:.2f}, Avg100: {avg_reward_100:.2f}, "
                          f"Replay size: {len(replay_buffer)}, "
                          f"Epsilon: {agent.epsilon:.4f}")
                
                except Exception as e:
                    print(f"\nError during evaluation: {e}")
                    traceback.print_exc()

            # 每save_freq步保存一次模型
            if total_steps % save_freq < collection_steps and total_steps > 0:
                try:
                    # 先保存模型，即使后面的操作失败也能恢复
                    model_path = os.path.join(output_path, "models", f"dqn_model_{total_steps}.pt")
                    torch.save(agent.policy_net.state_dict(), model_path)
                    print(f"\nModel saved to {model_path}")
                    
                    # 保存训练数据到CSV
                    data_path = os.path.join(output_path, f"training_data_{total_steps}.csv")
                    with open(data_path, 'w') as f:
                        f.write("step,reward,epsilon,loss\n")
                        for idx, (step, reward) in enumerate(zip(episode_steps, episode_rewards)):
                            if idx < len(epsilon_history) and idx < len(loss_history):
                                f.write(f"{step},{reward},{epsilon_history[idx]},{loss_history[idx]}\n")
                            else:
                                f.write(f"{step},{reward},N/A,N/A\n")
                    
                    # 记录一个样本视频（但频率较低，避免视频录制错误影响训练）
                    if total_steps % (save_freq * 5) < collection_steps and total_steps > 0 and ENABLE_VIDEO_RECORDING:
                        video_path = os.path.join(output_path, "videos")
                        record_game_video(agent, video_path, f"gameplay_{total_steps}")
                
                except Exception as e:
                    print(f"\nError saving checkpoint: {e}")
                    traceback.print_exc()
        
        # 检查是否达到总训练步数目标
        if total_steps >= total_steps_target:
            print(f"\nReached target of {total_steps_target} steps. Training complete.")
            break

    # ========== 训练结束, 保存最终模型并绘制图表 ==========
    end_time = time.time()
    total_training_time = end_time - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Total steps: {total_steps}, Episodes: {episode_count}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # 保存最终模型
    try:
        final_model_path = os.path.join(output_path, "models", "dqn_model_final.pt")
        torch.save(agent.policy_net.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # 录制最终游戏视频
        if ENABLE_VIDEO_RECORDING:
            final_video_path = os.path.join(output_path, "videos")
            final_reward = record_game_video(agent, final_video_path, "final_gameplay")
            print(f"Final gameplay video recorded. Reward: {final_reward}")
        else:
            print("Video recording disabled. Final gameplay video not recorded.")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    # 绘制最终训练图表
    try:
        plt.figure(figsize=(15, 12))
        
        # 1. 回合奖励曲线
        plt.subplot(2, 2, 1)
        plt.title("Episode Rewards")
        if episode_rewards:
            plt.plot(episode_steps, episode_rewards, 'b-', alpha=0.3, label='Individual Episodes')
            if len(episode_rewards) >= 10:
                moving_avg_10 = pd.Series(episode_rewards).rolling(10).mean()
                plt.plot(episode_steps[:len(moving_avg_10)], moving_avg_10, 'r-', label='10-Ep Moving Avg')
            if len(episode_rewards) >= 100:
                moving_avg_100 = pd.Series(episode_rewards).rolling(100).mean()
                plt.plot(episode_steps[:len(moving_avg_100)], moving_avg_100, 'g-', label='100-Ep Moving Avg')
            plt.legend()
        plt.grid(True)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        
        # 2. 评估奖励曲线
        plt.subplot(2, 2, 2)
        plt.title("Evaluation Rewards")
        eval_steps = [i*eval_freq for i in range(len(mean_rw_history))]
        plt.plot(eval_steps, mean_rw_history, 'go-')
        plt.grid(True)
        plt.xlabel("Steps")
        plt.ylabel("Eval Reward (avg of 3 games)")
        
        # 3. 训练损失曲线 (moving average)
        plt.subplot(2, 2, 3)
        plt.title("Training Loss (Smooth L1)")
        if len(loss_history) > 100:
            plt.plot(moving_average(loss_history, span=100, min_periods=100))
        else:
            plt.plot(loss_history)
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        
        # 4. Epsilon 随训练的变化
        plt.subplot(2, 2, 4)
        plt.title("Exploration Rate (Epsilon)")
        epsilon_steps = [i*collection_steps for i in range(len(epsilon_history))]
        plt.plot(epsilon_steps, epsilon_history)
        plt.grid(True)
        plt.xlabel("Steps")
        plt.ylabel("Epsilon")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "final_training_summary.png"), dpi=200)
        print(f"Final training summary plot saved.")
    except Exception as e:
        print(f"Error generating final plots: {e}")
    
    print("\nTraining session complete. All data saved to:", output_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()