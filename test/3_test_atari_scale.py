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
        # 创建 Breakout-v5 环境
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
    pygame.display.set_caption("Breakout-v5 - 用方向键控制拍板 (按 Q 退出)")
    # 固定窗口大小为原始像素的4倍
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
def run_game(env, screen, scale_factor=4):
    """
    运行主游戏循环。
    游戏内部逻辑使用原始像素，不做任何裁剪或变形；
    仅在渲染到屏幕时，将图像按比例放大4倍显示。
    """
    ACTIONS = {
        pygame.K_LEFT: 3,   # 向左移动
        pygame.K_RIGHT: 2,  # 向右移动
        pygame.K_SPACE: 1,  # 发球
    }

    done = False
    frame_delay = 0.02  # 控制帧率（秒）, 0.02=50fps, 0.05=20fps

    # 重置环境并获取初始观察
    observation, info = env.reset()
    print("初始 observation.shape =", observation.shape)
    # observation 的 shape 通常为 (210, 160, 3)
    original_height, original_width, _ = observation.shape

    while not done:
        action = 0  # 默认不动

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                elif event.key in ACTIONS:
                    action = ACTIONS[event.key]

        # 执行动作，游戏逻辑依然使用原始像素
        observation, reward, done, truncated, info = env.step(action)

        # 方法1：先直接放大 observation，再转置
        target_width = original_width * scale_factor    # 160 * 4 = 640
        target_height = original_height * scale_factor  # 210 * 4 = 840
        # cv2.resize 参数顺序为 (宽, 高)
        frame = cv2.resize(observation, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        # cv2.resize 返回的 shape 为 (840, 640, 3)，需要转置为 (640,840,3) 供 pygame 使用
        frame = np.transpose(frame, (1, 0, 2))

        # 方法2：如果你更喜欢先转置再放大，也可以这样做：
        # frame = np.transpose(observation, (1, 0, 2))       # (160,210,3)
        # frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        # frame = np.transpose(frame, (1, 0, 2))  # 恢复到 (640,840,3)

        # 显示图像
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

    # 窗口尺寸固定为原始像素的4倍
    scale_factor = 4
    window_width = original_width * scale_factor   # 160 * 4 = 640
    window_height = original_height * scale_factor # 210 * 4 = 840

    # 初始化 Pygame 窗口
    screen = initialize_pygame(window_width, window_height)

    # 运行游戏主循环
    run_game(env, screen, scale_factor)
