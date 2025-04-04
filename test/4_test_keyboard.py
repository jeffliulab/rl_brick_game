import os
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import time
import sys
import pygame  # 用于捕捉键盘输入

# =============================
# 配置与初始化，ROM读取验证等
# =============================
def initialize_environment():
    """
    初始化 Gymnasium 环境，并检查必要的配置。
    这里只使用内置跳帧的版本："ALE/Breakout-v5"。
    返回创建好的环境对象。
    """
    print(f"Gymnasium version: {gym.__version__}")
    print(f"ALE-Py version: {ale_py.__version__}")

    # 检查 ROM_PATH 是否配置
    rom_path = os.getenv("ROM_PATH")
    print(f"ROM_PATH: {rom_path}")

    # 列出已注册的环境
    print(f"已注册的环境列表：{list(gym.envs.registry.keys())}")

    try:
        # 只使用内置跳帧版本
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        print("环境加载成功！")
        return env
    except Exception as e:
        print(f"环境加载失败：{e}")
        sys.exit(1)

# =============================
# Pygame 初始化
# =============================
def initialize_pygame(window_width, window_height):
    """
    初始化 Pygame 窗口和设置。
    返回 Pygame 窗口对象。
    """
    pygame.init()
    pygame.display.set_caption("Breakout - 用方向键控制拍板 (按 Q 退出)")
    # 固定窗口大小为原始像素的 scale_factor 倍
    screen = pygame.display.set_mode((window_width, window_height))
    
    print("🎮 控制方式：")
    print("⬅️ 左箭头：向左移动")
    print("➡️ 右箭头：向右移动")
    print("⏺ 空格键：发球")
    print("🔲 Q键：退出游戏")

    return screen

# =============================
# 游戏主循环
# =============================
def run_game(env, screen, scale_factor=4, frame_skip=1):
    """
    运行主游戏循环。
    - 如果 frame_skip > 1，则在同一时间步内重复执行相同动作（不过环境内部本身已包含跳帧机制）。
    - 图像渲染时采用 INTER_NEAREST 插值，只放大像素而不生成新像素。
    - 使用 pygame.key.get_pressed() 模拟 agent 在每个时间步选择动作。
    """
    # 定义键盘对应的动作（注意：环境内置跳帧，动作效果会连续执行多帧）
    ACTIONS = {
        pygame.K_LEFT: 3,   # 向左移动
        pygame.K_RIGHT: 2,  # 向右移动
        pygame.K_SPACE: 1,  # 发球
    }

    done = False
    frame_delay = 0.02  # 控制帧率，例如 0.02 秒对应约 50fps

    # 重置环境并获取初始观察
    observation, info = env.reset()
    print("初始 observation.shape =", observation.shape)
    original_height, original_width, _ = observation.shape

    while not done:
        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # 使用连续按键检测，确保每个时间步都能检测到按键
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            done = True

        # 模拟 agent 每个时间步选择动作
        if keys[pygame.K_LEFT]:
            action = ACTIONS[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]:
            action = ACTIONS[pygame.K_RIGHT]
        elif keys[pygame.K_SPACE]:
            action = ACTIONS[pygame.K_SPACE]
        else:
            action = 0  # 默认无动作

        total_reward = 0
        # 如果 frame_skip > 1，则在当前时间步内重复执行相同动作
        for _ in range(frame_skip):
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done:
                break

        # 渲染：先用 INTER_NEAREST 放大图像，确保仅复制像素点
        target_width = original_width * scale_factor
        target_height = original_height * scale_factor
        frame = cv2.resize(observation, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        # cv2.resize 返回的 shape 为 (target_height, target_width, 3)
        # 转置为 (target_width, target_height, 3) 以适应 pygame 显示
        frame = np.transpose(frame, (1, 0, 2))

        # 在 Pygame 窗口中显示图像
        surface = pygame.surfarray.make_surface(frame)
        screen.fill((0, 0, 0))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        time.sleep(frame_delay)

    env.close()
    pygame.quit()
    sys.exit()

# =============================
# 主程序入口
# =============================
if __name__ == "__main__":
    env = initialize_environment()

    # 获取初始 observation 的原始像素尺寸
    observation, info = env.reset()
    print("reset 后的 observation.shape =", observation.shape)
    original_height, original_width, _ = observation.shape

    # 固定窗口尺寸为原始像素的 scale_factor 倍
    scale_factor = 4
    window_width = original_width * scale_factor
    window_height = original_height * scale_factor

    screen = initialize_pygame(window_width, window_height)

    # 因为环境内置跳帧，若希望键盘动作完全模拟 agent，则 frame_skip 设为 1
    run_game(env, screen, scale_factor=scale_factor, frame_skip=1)
