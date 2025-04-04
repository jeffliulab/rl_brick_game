#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例：使用 DQN 在本地环境中训练 Atari Breakout (ALE/Breakout-v5)
主要依赖：
  - gymnasium
  - ale_py
  - opencv-python
  - torch
请先安装：
  pip install gymnasium ale-py opencv-python torch
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Gymnasium & ALE-Py =====
import gymnasium as gym
import ale_py

# ===== PyTorch =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===== 其他辅助库 =====
import cv2
from tqdm import trange
from collections import deque
from pandas import DataFrame

# ========== 1. 全局配置：随机种子、设备选择等 ==========

# 设置随机种子，保证实验可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# 检查 GPU，如果支持则使用，否则 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前 PyTorch 使用的设备:", device)

# 用于绘图曲线的移动平均
def moving_average(x, span, **kw):
    return (
        DataFrame({"x": np.asarray(x)})
        .x.ewm(span=span, **kw)
        .mean()
        .values
    )


# ========== 2. 初始化环境的函数 ==========

def initialize_environment(render_mode="rgb_array", frameskip=4):
    """
    初始化并返回一个 "ALE/Breakout-v5" 环境
    1) 打印 Gymnasium / ALE-Py 版本
    2) 显示 ROM_PATH（若有设置）
    3) 列出注册环境
    4) 创建 "ALE/Breakout-v5" 并返回
    """
    # print(f"Gymnasium 版本: {gym.__version__}")
    # print(f"ALE-Py 版本: {ale_py.__version__}")

    rom_path = os.getenv("ROM_PATH")
    # print(f"ROM_PATH: {rom_path}")

    # print("已注册的环境列表：")
    # print(list(gym.envs.registry.keys()))

    try:
        # 创建 Atari Breakout 环境
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode=render_mode,
            frameskip=frameskip
        )
        print("环境加载成功！")
        return env
    except Exception as e:
        print(f"环境加载失败：{e}")
        sys.exit(1)


# ========== 3. 预处理包装器（裁剪/缩放/灰度）与帧缓冲包装器 ==========

class PreprocessAtari(gym.ObservationWrapper):
    """
    对 Atari 画面进行预处理：
      - 裁剪上下无用区域
      - 缩放至 (84, 84)
      - 转为灰度
      - 归一化到 [0,1]
    """
    def __init__(self, env):
        super().__init__(env)
        self.img_size = (84, 84)

        # 修改观察空间为 84x84 单通道
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.img_size[0], self.img_size[1], 1),
            dtype=np.float32
        )

    def observation(self, obs):
        # 裁剪掉上面 34 和下面 16 像素
        obs = obs[34:-16, :, :]
        # 调整大小 => (84,84)
        obs = cv2.resize(obs, self.img_size)
        # 转为灰度（单通道）
        obs = obs.mean(axis=-1, keepdims=True)
        # 归一化
        obs = obs.astype(np.float32) / 255.0
        return obs


class FrameBuffer(gym.Wrapper):
    """
    帧缓冲：存储最近 n_frames 帧画面，使网络可以感知运动信息。
    支持 'pytorch' 或 'tensorflow' 形式：
      - pytorch: [C, H, W]
      - tensorflow: [H, W, C]
    """
    def __init__(self, env, n_frames=4, dim_order='pytorch'):
        super().__init__(env)
        self.dim_order = dim_order
        height, width, channels = env.observation_space.shape

        if dim_order == 'pytorch':
            # PyTorch => [C, H, W]
            self.frame_shape = (channels, height, width)
            obs_shape = (channels * n_frames, height, width)
        elif dim_order == 'tensorflow':
            # TensorFlow => [H, W, C]
            self.frame_shape = (height, width, channels)
            obs_shape = (height, width, channels * n_frames)
        else:
            raise ValueError('dim_order 只能是 "pytorch" 或 "tensorflow"')

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )

        self.n_frames = n_frames
        self.framebuffer = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        # 调用原始 env.reset()
        obs, info = self.env.reset(**kwargs)
        # 缓冲清空
        self.framebuffer = np.zeros_like(self.framebuffer)
        # 更新缓冲区
        self.update_buffer(obs)
        return self.framebuffer, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or truncated
        self.update_buffer(obs)
        return self.framebuffer, reward, done, info

    def update_buffer(self, obs):
        """
        将新的一帧 obs 放到最前面，移除最老的一帧
        """
        if self.dim_order == 'pytorch':
            # [H, W, C] => [C, H, W]
            obs = np.transpose(obs, (2, 0, 1))
            old = self.framebuffer[:-self.frame_shape[0], :, :]
            self.framebuffer = np.concatenate([obs, old])
        else:
            c = self.frame_shape[-1]  # 每帧通道数
            old = self.framebuffer[:, :, :-c]
            self.framebuffer = np.concatenate([obs, old], axis=-1)


# ========== 4. DQN 模型、经验回放、代理类 ==========

class DQN(nn.Module):
    """
    使用深度卷积网络，对状态 s (4帧拼接后的 84x84) 进行特征提取，
    输出对应 n_actions 个 Q值
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # 构造一个空的输入，推断卷积输出大小
        test_inp = torch.zeros(1, *shape)
        out = self.conv(test_inp)
        return int(np.prod(out.size()))

    def forward(self, x):
        # x shape: [batch_size, C, H, W]
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


class ReplayBuffer:
    """
    经验回放缓冲区，存储 (s, a, r, s_next, done)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        # 为避免 PyTorch 警告“list of numpy.ndarrays is slow”，统一合并为 np.array
        s = np.array(s, dtype=np.float32)
        a = np.array(a, dtype=np.int64)
        r = np.array(r, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        # 转为 Tensor 并放到对应 device
        s_t = torch.from_numpy(s).to(device)
        a_t = torch.from_numpy(a).to(device)
        r_t = torch.from_numpy(r).to(device)
        s_next_t = torch.from_numpy(s_next).to(device)
        done_t = torch.from_numpy(done).to(device)

        return s_t, a_t, r_t, s_next_t, done_t


class DQNAgent:
    """
    DQN 代理类：包含
      - policy_net (在线网络)
      - target_net (目标网络)
      - epsilon-greedy 选动作
    """
    def __init__(self, state_shape, n_actions, epsilon=1.0):
        self.epsilon = epsilon
        self.n_actions = n_actions

        # 策略网络
        self.policy_net = DQN(state_shape, n_actions).to(device)
        # 目标网络
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.update_target_network()

        # 目标网络不需要梯度
        for p in self.target_net.parameters():
            p.requires_grad = False

    def update_target_network(self):
        """
        将 policy_net 的参数复制到 target_net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_qvalues(self, states):
        """
        给定一批状态 states (np.array)，返回 Q(s,·)
        states shape: [batch_size, C, H, W]
        """
        with torch.no_grad():
            inp = torch.from_numpy(states).float().to(device)
            qvals = self.policy_net(inp)
            return qvals.cpu().numpy()

    def sample_actions(self, qvalues):
        """
        根据 epsilon-greedy 在 qvalues 中选动作
        qvalues shape: [batch_size, n_actions]
        """
        batch_size = qvalues.shape[0]
        # 纯随机动作
        random_actions = np.random.choice(self.n_actions, size=batch_size)
        # 选择 Q 值最大的动作
        best_actions = qvalues.argmax(axis=1)

        explore = np.random.rand(batch_size) < self.epsilon
        return np.where(explore, random_actions, best_actions)


# ========== 5. 训练、评估函数 ==========

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """
    让 agent 在 env 中玩 n_games 局，并计算平均总回报
    若 greedy=True，则每步 action 用 argmax(Q)，否则用 ε-贪心
    """
    rewards = []
    for _ in range(n_games):
        obs, info = env.reset()
        ep_reward = 0
        for _ in range(t_max):
            # 构造 batch=[1, ...] 的状态
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
    与环境交互 n_steps 步，将产生的 (s, a, r, s_next, done) 存入回放缓冲区。
    若在过程中 done=True，则自动 reset 并继续剩余步数。
    返回这 n_steps 中累积的奖励。
    """
    # 先拿到当前状态
    # 注意：FrameBuffer里 self.framebuffer 存的就是当前组合帧
    s = env.framebuffer
    reward_sum = 0.0

    for _ in range(n_steps):
        # 批量方式 => [1, C, H, W]
        state_batch = np.array([s], dtype=np.float32)
        qvals = agent.get_qvalues(state_batch)
        action = agent.sample_actions(qvals)[0]

        s_next, reward, done, info = env.step(action)

        # 写入回放缓冲
        replay_buffer.add(s, action, reward, s_next, float(done))

        reward_sum += reward
        if done:
            # 回合结束，则重新 reset
            s_reset, _ = env.reset()
            s = s_reset
        else:
            s = s_next

    return reward_sum


def train_on_batch(agent, optimizer, replay_buffer, batch_size=64, gamma=0.99):
    """
    从 replay_buffer 里采样一个 batch 做一次梯度更新
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # 计算当前 Q(s,a)
    qvals = agent.policy_net(states)  # [batch, n_actions]
    current_q = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 计算目标 Q(s,a)
    # target = r + gamma * max Q(s',·) * (1 - done)
    with torch.no_grad():
        qvals_next = agent.target_net(next_states)
        max_q_next = qvals_next.max(dim=1)[0]
        target_q = rewards + gamma * max_q_next * (1 - dones)

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()

    # 可选：对梯度做裁剪，避免梯度爆炸
    for param in agent.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    return loss.item()


# ========== 6. main() 主函数，将上述流程整合在一起 ==========

def main():
    """
    主函数：
      1) 初始化环境并包装
      2) 创建 DQNAgent 等
      3) 先 reset，再收集经验并训练
      4) 定期评估、保存模型
      5) 最终录制一局视频
      6) 绘制训练曲线
    """
    # (1) 初始化原始环境
    raw_env = initialize_environment(render_mode="rgb_array", frameskip=4)

    # (2) 包上预处理与帧缓冲
    env = PreprocessAtari(raw_env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')

    # 查看动作空间
    n_actions = env.action_space.n
    print("动作空间大小:", n_actions)

    # 状态空间形状，如 (4, 84, 84)
    state_shape = env.observation_space.shape

    # (3) 创建 DQNAgent、优化器、回放缓冲
    agent = DQNAgent(state_shape, n_actions, epsilon=1.0)
    optimizer = optim.Adam(agent.policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=70000)

    # 必须先 reset 再 step，避免 "Cannot call env.step() before calling env.reset()"
    obs, info = env.reset()

    # 先预填充回放缓冲
    print("回放缓冲填充中，请稍候...")
    play_and_record(agent, env, replay_buffer, n_steps=10000)

    # (4) 训练循环中的一些统计
    mean_rw_history = []
    loss_history = []

    # 一些超参数
    gamma = 0.99
    batch_size = 64
    target_update_freq = 500
    epsilon_decay = 0.999
    min_epsilon = 0.01
    eval_freq = 5000
    save_freq = 20000
    num_iterations = 100000

    print("开始训练 DQN ...")
    for i in trange(num_iterations):
        # 每个 iteration: 先收集一些新经验
        play_and_record(agent, env, replay_buffer, n_steps=10)

        # 再用这些经验训练一个 batch
        loss = train_on_batch(agent, optimizer, replay_buffer, batch_size, gamma)
        loss_history.append(loss)

        # 定期更新 target_net 并衰减 epsilon
        if i % target_update_freq == 0:
            agent.update_target_network()
            agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)

        # 定期评估
        if i % eval_freq == 0:
            eval_env = initialize_environment(render_mode="rgb_array", frameskip=4)
            eval_env = PreprocessAtari(eval_env)
            eval_env = FrameBuffer(eval_env, n_frames=4, dim_order='pytorch')
            obs_eval, _ = eval_env.reset()

            mean_reward = evaluate(eval_env, agent, n_games=3, greedy=True)
            mean_rw_history.append(mean_reward)
            print(f"迭代 {i}, 平均奖励: {mean_reward:.2f}, "
                  f"缓冲区大小: {len(replay_buffer)}, "
                  f"Epsilon: {agent.epsilon:.3f}")

        # 定期保存模型
        if i % save_freq == 0 and i > 0:
            torch.save(agent.policy_net.state_dict(), f"dqn_model_atari_weights_{i}.pt")

    # (5) 训练结束后，保存最终模型
    torch.save(agent.policy_net.state_dict(), "dqn_model_atari_weights_final.pt")

    # 将 epsilon 设为 0 做最终评估
    agent.epsilon = 0

    # 录制一局游戏视频
    video_env = gym.wrappers.RecordVideo(
        initialize_environment(render_mode="rgb_array"),
        video_folder="videos",
        episode_trigger=lambda e: True
    )
    # 如果想录制原始画面，可以不包预处理；若想录制灰度图，可以包
    video_env = PreprocessAtari(video_env)
    video_env = FrameBuffer(video_env, n_frames=4, dim_order='pytorch')
    video_obs, _ = video_env.reset()

    final_rewards = [evaluate(video_env, agent, n_games=1, greedy=True)]
    video_env.close()

    print(f"训练结束，最终平均奖励: {np.mean(final_rewards):.2f}")

    # (6) 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("评估时的平均奖励 (每隔一段评估一次)")
    plt.plot(mean_rw_history)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("训练损失 (平滑后)")
    plt.plot(moving_average(loss_history, span=100, min_periods=100))
    plt.grid()

    plt.savefig("dqn_training_curve.png")
    plt.show()


# 以脚本方式执行
if __name__ == "__main__":
    main()
