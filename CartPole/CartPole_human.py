'''
人操控
'''

import gymnasium as gym
import time
import numpy as np
import torch
from torch import nn
import random
import torch.nn.utils as torch_utils

env = gym.make('CartPole-v1', render_mode="human")

train_epoch = 100


def set_reproducible(seed=42):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 禁用多线程
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_reproducible()

for i in range(train_epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0

    for j in range(500):
        # 根据模型选取动作
        old_state_tensor = torch.tensor(old_state)
        action = int(input())
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward

        if terminated or truncated:
            if terminated:  # 没到500才学，到500了容易学坏掉
                pass
            print("中止或者结束")
            print("i: ", i)
            print("all_reward: ", all_reward)
            print("="*30)
            break

        old_state = new_state
