import gymnasium as gym
import time
import numpy as np
import torch
from torch import nn
import random

# env = gym.make("FrozenLake-v1", render_mode="human")rl
env = gym.make('FrozenLake-v1',
               desc=None,
               map_name="4x4",
               is_slippery=False,   # 湖是不是滑的
               render_mode="human",
               )

epoch = 500
gama = 0.99


def get_row_col(idx, cols=4):
    return (idx // cols + 1, idx % cols + 1)


model = nn.Linear(in_features=16, out_features=4, bias=False)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

dire_map = ['左', '下', '右', '上']


def print_all_Q():
    from pprint import pprint
    all_state = torch.arange(16)
    all_state_one_hot = nn.functional.one_hot(all_state).float()
    with torch.no_grad():
        all_q = model(all_state_one_hot)

    for idx, line in enumerate(all_q):
        row, col = get_row_col(idx)
        print("{}行，{}列, 最大: {}, {}".format(
            row, col, dire_map[torch.argmax(line)], line.tolist()))
        if col == 4:
            print("="*20)


for i in range(epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0
    threshold = 1 / ((i / 50) + 10)
    for j in range(99):
        # 根据模型选取动作
        # torch.nn.functional.one_hot(torch.tensor([10]), num_classes=20)
        old_state_one_hot = nn.functional.one_hot(
            torch.tensor(old_state), num_classes=16)
        with torch.no_grad():
            action_distribution = model(old_state_one_hot.float())
        action = torch.argmax(action_distribution).item()
        # 如果小于门槛，就随机sample一个动作
        if random.random() < threshold:
            action = random.randint(0, env.action_space.n - 1)
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward
        # 新状态下, 根据当前策略最大概率的动作max_a对应的最大价值
        new_state_one_hot = nn.functional.one_hot(
            torch.tensor(new_state), num_classes=16)
        with torch.no_grad():
            next_action_distribution = model(new_state_one_hot.float())
        max_a = torch.max(next_action_distribution).item()

        # 反向传播
        model_output = model(old_state_one_hot.float())
        # 需要对齐的Q
        ground_truth = reward + gama * max_a
        loss = loss_fn(model_output[action], torch.tensor(ground_truth))
        loss.backward()
        optimizer.step()

        if terminated or truncated:
            print("old_state: ", get_row_col(old_state))
            print("new_state: ", get_row_col(new_state))
            break
        old_state = new_state
    if all_reward != 0:
        print_all_Q()
    print("epoch: ", i)
    print("all_reward: ", all_reward)

print_all_Q()
