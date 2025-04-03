import os
import gymnasium as gym
import ale_py

# 确认版本
print(f"Gymnasium version: {gym.__version__}")
print(f"ALE-Py version: {ale_py.__version__}")

# 确认路径配置
rom_path = os.getenv("ROM_PATH")
print(f"ROM_PATH: {rom_path}")

# 列出所有可用的环境
print(f"已注册的环境列表：{list(gym.envs.registry.keys())}")


import gymnasium as gym
import numpy as np
import cv2
import time
import sys
import pygame  # 用于捕捉键盘输入

# 初始化 Pygame
pygame.init()
pygame.display.set_caption("Breakout-v5 - 用方向键控制拍板 (按 Q 退出)")

# 创建 Breakout-v5 环境
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
observation, info = env.reset()

# 设置游戏参数
done = False
frame_delay = 0.02  # 控制帧率（秒）

# 动作映射
ACTIONS = {
    pygame.K_LEFT: 3,  # 向左移动
    pygame.K_RIGHT: 2,  # 向右移动
    pygame.K_SPACE: 1,  # 发球
}

# 创建 Pygame 窗口
screen_width, screen_height = observation.shape[1], observation.shape[0]
screen = pygame.display.set_mode((screen_width, screen_height))

print("🎮 控制方式：")
print("⬅️ 左箭头：向左移动")
print("➡️ 右箭头：向右移动")
print("⏺ 空格键：发球")
print("🔲 Q键：退出游戏")

while not done:
    action = 0  # 默认动作：不移动

    # 处理 Pygame 事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                done = True
                break
            elif event.key in ACTIONS:
                action = ACTIONS[event.key]

    # 执行动作
    observation, reward, done, truncated, info = env.step(action)

    # 将 OpenAI Gym 提供的图像数据转换为 Pygame 的 Surface
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    frame = np.transpose(frame, (1, 0, 2))  # 转置矩阵，使其与 Pygame 显示格式匹配
    surface = pygame.surfarray.make_surface(frame)

    # 显示在 Pygame 窗口中
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # 控制帧率
    time.sleep(frame_delay)

# 清理资源
env.close()
pygame.quit()
sys.exit()
