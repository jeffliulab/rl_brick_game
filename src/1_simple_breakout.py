import pygame
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 区域 1：Breakout 游戏环境
# =============================================================================
class BreakoutEnv:
    """
    离散版 Breakout 游戏环境
    -----------------------------------
    模拟一个简化版的 Breakout 游戏，采用10x10的离散网格：
      - 挡板固定在最底行，宽度为3格，可左右移动
      - 球从挡板上方起始，具有固定初始竖直向上速度及随机水平速度
      - 砖块只分布在最顶行，用一个10位二进制数表示（每一位对应一个砖块）
    提供的接口：
      - reset(): 重置环境并返回初始状态
      - step(action): 执行动作并返回 (next_state, reward, done)
      - get_state(): 获取当前环境状态（用于代理观察）
    """
    def __init__(self):
        # 网格尺寸设置
        self.cols = 10
        self.rows = 10
        # 挡板参数：宽度为3格，固定在最底行
        self.paddle_width = 3
        self.paddle_y = self.rows - 1
        # 砖块初始状态：10个砖块全存在，使用二进制表示（2^10 - 1）
        self.initial_bricks = (1 << 10) - 1

    def reset(self):
        """
        重置环境状态：
          - 挡板位置居中
          - 球的位置在挡板正上方，水平位置在挡板范围内随机取值
          - 球初始速度：竖直向上 (-1)；水平速度随机选择 -1 或 1
          - 砖块状态重置为全存在
        返回：初始状态 tuple (ball_x, ball_y, vx, vy, paddle_x, brick_count)
        """
        self.paddle_x = (self.cols - self.paddle_width) // 2
        self.ball_x = random.randint(self.paddle_x, self.paddle_x + self.paddle_width - 1)
        self.ball_y = self.paddle_y - 1
        self.vx = random.choice([-1, 1])
        self.vy = -1
        self.bricks = self.initial_bricks
        self.t = 0
        return self.get_state()

    def get_state(self):
        """
        获取当前环境状态，状态简化为：
          (ball_x, ball_y, vx, vy, paddle_x, brick_count)
        其中 brick_count 表示当前剩余砖块数量。
        """
        brick_count = bin(self.bricks).count("1")
        return (self.ball_x, self.ball_y, self.vx, self.vy, self.paddle_x, brick_count)

    def step(self, action):
        """
        根据动作更新环境状态，并计算即时奖励。
        动作定义：
          0: 不动
          1: 向左移动挡板一格
          2: 向右移动挡板一格

        奖励规则：
          - 每步基础惩罚：-0.1
          - 撞击砖块：+5（同时消除砖块，改变球的方向）
          - 成功接到球：+1（挡板接住球，反弹球）
          - 游戏失败（球掉出底部）：-10
          - 游戏胜利（砖块全部消除）：+10

        返回：
          next_state: 更新后的状态
          reward: 本步获得的奖励
          done: 是否终止（True 表示游戏结束）
        """
        # 更新挡板位置（确保在边界内）
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.cols - self.paddle_width, self.paddle_x + 1)

        reward = -0.1  # 每步基础惩罚

        # 计算球的新位置
        next_ball_x = self.ball_x + self.vx
        next_ball_y = self.ball_y + self.vy

        # 处理左右墙壁碰撞
        if next_ball_x < 0:
            next_ball_x = 0
            self.vx = -self.vx
        elif next_ball_x >= self.cols:
            next_ball_x = self.cols - 1
            self.vx = -self.vx

        # 处理顶部碰撞
        if next_ball_y < 0:
            next_ball_y = 0
            self.vy = -self.vy

        # 检查砖块碰撞（仅当球到达最顶行 row==0 时）
        if next_ball_y == 0:
            brick_bit = 1 << next_ball_x
            if self.bricks & brick_bit:
                self.bricks = self.bricks & (~brick_bit)  # 消除砖块
                self.vy = -self.vy
                reward += 5  # 撞砖块奖励
                next_ball_y = 0  # 保持在顶行

        # 检查挡板碰撞（挡板位于最底行）
        if next_ball_y == self.paddle_y:
            if self.paddle_x <= next_ball_x < self.paddle_x + self.paddle_width:
                self.vy = -self.vy
                reward += 1  # 成功接到球奖励
                next_ball_y = self.paddle_y - 1

        # 更新球的位置
        self.ball_x = next_ball_x
        self.ball_y = next_ball_y
        self.t += 1

        # 判断游戏结束条件：球越过底部或砖块全部消除
        done = False
        if self.ball_y >= self.rows:
            done = True
        if self.bricks == 0:
            done = True

        # 游戏结束时给予最终奖励
        if done:
            if self.ball_y >= self.rows:
                reward = -10  # 失败
            elif self.bricks == 0:
                reward = 10   # 胜利

        return self.get_state(), reward, done

# =============================================================================
# 区域 2：游戏图形展示与接口封装
# =============================================================================
class BreakoutRenderer:
    """
    使用 Pygame 展示 Breakout 游戏图形界面
    -------------------------------------------------
    该类负责将环境状态通过图形界面展示出来，同时提供接口使强化学习代理
    可以通过环境的状态进行决策。
    
    参数：
      env: BreakoutEnv 环境实例
      q_agent: 强化学习代理（可选）。若提供，则在每一帧中根据代理策略选择动作；
               否则默认不移动（动作 0）。
      grid_size: 每个网格的像素大小（默认为50）
      fps: 演示时的帧率（默认每秒5帧）
    """
    def __init__(self, env, q_agent=None, grid_size=50, fps=5):
        pygame.init()
        self.env = env
        self.q_agent = q_agent
        self.grid_size = grid_size
        self.cols = env.cols
        self.rows = env.rows
        self.width = self.cols * grid_size
        self.height = self.rows * grid_size
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Learning Breakout Demo")
        self.clock = pygame.time.Clock()

    def run(self):
        """
        主运行循环：
          - 根据强化学习代理选择动作（若提供代理，则使用其贪婪策略）
          - 更新环境状态
          - 渲染当前状态
          - 检查退出条件
        """
        state = self.env.reset()
        running = True
        while running:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 采用强化学习代理的策略（贪婪策略），若未提供代理则默认动作 0
            if self.q_agent is not None:
                action = self.q_agent.get_action(state, greedy=True)
            else:
                action = 0

            state, reward, done = self.env.step(action)
            self.render()

            if done:
                running = False

        pygame.time.wait(3000)
        pygame.quit()

    def render(self):
        """
        绘制当前游戏状态：
          - 绘制第一行的砖块（红色方块）
          - 绘制挡板（蓝色矩形）
          - 绘制球（红色圆形）
        """
        self.screen.fill((255, 255, 255))
        # 绘制砖块（仅第一行有砖块）
        for i in range(self.cols):
            if self.env.bricks & (1 << i):
                rect = pygame.Rect(i * self.grid_size, 0, self.grid_size, self.grid_size)
                pygame.draw.rect(self.screen, (255, 0, 0), rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        # 绘制挡板
        paddle_rect = pygame.Rect(
            self.env.paddle_x * self.grid_size,
            self.env.paddle_y * self.grid_size,
            self.env.paddle_width * self.grid_size,
            self.grid_size
        )
        pygame.draw.rect(self.screen, (0, 0, 255), paddle_rect)
        # 绘制球
        center = (int(self.env.ball_x * self.grid_size + self.grid_size/2),
                  int(self.env.ball_y * self.grid_size + self.grid_size/2))
        radius = self.grid_size // 2
        pygame.draw.circle(self.screen, (255, 0, 0), center, radius)
        pygame.display.flip()

# =============================================================================
# 区域 3：强化学习部分（Q-Learning代理）
# =============================================================================
class QLearningAgent:
    """
    Q-Learning 强化学习代理
    -------------------------------------------------
    该代理在 Breakout 环境中使用 Q-Learning 算法进行学习，维护一个 Q 表，
    键为环境状态元组，值为对应动作（0、1、2）的 Q 值列表。
    
    主要接口：
      - get_action(state, epsilon=0.0, greedy=False): 根据 epsilon-greedy 策略选择动作
      - update(state, action, reward, next_state): 使用 Q-Learning 公式更新 Q 值
      - train(env, num_episodes): 在给定环境中训练代理，并记录训练过程中的奖励和成功率
    """
    def __init__(self, alpha=0.1, gamma=0.99, initial_epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.Q = {}  # Q 表：状态 -> [Q(a=0), Q(a=1), Q(a=2)]
        self.episode_rewards = []  # 每个 episode 的累计奖励
        self.episode_success = []  # 每个 episode 是否成功（砖块全消除为成功）

    def get_Q(self, state):
        """
        获取状态 state 对应的 Q 值列表，
        如果状态未在 Q 表中出现，则初始化为 [0, 0, 0]
        """
        if state not in self.Q:
            self.Q[state] = [0, 0, 0]
        return self.Q[state]

    def get_action(self, state, epsilon=0.0, greedy=False):
        """
        根据 epsilon-greedy 策略选择动作：
          - 若 greedy=True，则总是选择最大 Q 值对应动作（贪婪策略）
          - 否则，根据 epsilon 随机选择探索动作或贪婪动作
        """
        if greedy:
            return int(np.argmax(self.get_Q(state)))
        else:
            if random.random() < epsilon:
                return random.randint(0, 2)
            else:
                return int(np.argmax(self.get_Q(state)))

    def update(self, state, action, reward, next_state):
        """
        根据 Q-Learning 更新公式更新 Q 值：
          Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',:)) - Q(s,a))
        """
        best_next = max(self.get_Q(next_state))
        current = self.get_Q(state)[action]
        self.Q[state][action] = current + self.alpha * (reward + self.gamma * best_next - current)

    def train(self, env, num_episodes=10000):
        """
        在环境 env 中训练代理 num_episodes 次
        参数：
          env: BreakoutEnv 实例
          num_episodes: 训练的总回合数
        记录每个 episode 的累计奖励和成功情况（砖块全消除即视为成功）
        返回：
          Q 表, episode_rewards, episode_success
        """
        total_reward_accum = 0

        for episode in range(num_episodes):
            # 线性衰减 epsilon：从 initial_epsilon 逐渐衰减到 0
            epsilon = self.initial_epsilon * (1 - episode / num_episodes)
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(state, epsilon=epsilon)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                self.update(state, action, reward, next_state)
                state = next_state

            total_reward_accum += episode_reward
            self.episode_rewards.append(episode_reward)
            # 判断成功：砖块全部消除即为成功
            if env.bricks == 0:
                self.episode_success.append(1)
            else:
                self.episode_success.append(0)

            # 每1000个episode输出一次训练统计信息
            if (episode + 1) % 1000 == 0:
                avg_reward = total_reward_accum / 1000.0
                print(f"Episode {episode+1}/{num_episodes} - Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
                total_reward_accum = 0
        return self.Q, self.episode_rewards, self.episode_success

# =============================================================================
# 区域 4：绘制学习曲线
# =============================================================================
def plot_learning_curves(episode_rewards, episode_success, window=100):
    """
    绘制训练过程中学习曲线：
      - 左图：每个 episode 的奖励及滑动窗口平均
      - 右图：每个 episode 的成功率及滑动窗口平均
    参数：
      episode_rewards: 每个 episode 的累计奖励列表
      episode_success: 每个 episode 的成功标志列表（1表示成功，0表示失败）
      window: 用于计算滑动平均的窗口大小
    """
    episodes = np.arange(len(episode_rewards))
    rewards = np.array(episode_rewards)
    success = np.array(episode_success)

    # 定义滑动窗口平均函数
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    reward_avg = moving_average(rewards, window)
    success_rate = moving_average(success, window)

    plt.figure(figsize=(12, 5))

    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(episodes[window-1:], reward_avg, label=f"Moving Average (w={window})", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.legend()

    # 绘制成功率曲线
    plt.subplot(1, 2, 2)
    plt.plot(episodes[window-1:], success_rate, label=f"Success Rate (w={window})", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# 区域 5：主流程
# =============================================================================
def main():
    """
    主流程：
      1. 初始化游戏环境和 Q-Learning 代理
      2. 在环境中训练代理
      3. 使用训练好的代理进行图形演示
      4. 绘制训练过程中的学习曲线
    """
    # 创建游戏环境实例
    env = BreakoutEnv()

    # 初始化 Q-Learning 代理
    q_agent = QLearningAgent(alpha=0.1, gamma=0.99, initial_epsilon=0.2)

    # 在环境中训练代理（可调节 num_episodes 训练轮数）
    print("Training Q-learning agent...")
    Q, episode_rewards, episode_success = q_agent.train(env, num_episodes=100000)
    print("Training completed. Now demonstrating the learned policy...")

    # 使用 Pygame 图形界面演示训练好的策略
    renderer = BreakoutRenderer(env, q_agent=q_agent, grid_size=50, fps=5)
    renderer.run()

    # 绘制训练过程中的学习曲线
    print("Plotting learning curves...")
    plot_learning_curves(episode_rewards, episode_success, window=100)

if __name__ == "__main__":
    main()
