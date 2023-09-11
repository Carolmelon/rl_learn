'''
改编自：https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_DQN.py

暂时不太行
'''

import gymnasium as gym
import time
import numpy as np
import torch
from torch import nn
import random

env = gym.make('CartPole-v1', render_mode="human")

epoch = 10
gama = 0.99


class CartPoleDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=4, out_features=2, bias=False)

    def forward(self, x):
        return self.linear1(x)


class CartPoleDQN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=4, out_features=30, bias=True)
        self.linear3 = nn.Linear(in_features=30, out_features=2, bias=True)
        self.act_fn = nn.Tanh()
        # 初始化权重w为均值为0，方差为0.03
        nn.init.normal_(self.linear1.weight, mean=0, std=0.03)
        nn.init.normal_(self.linear3.weight, mean=0, std=0.03)

        # 初始化偏置b为常数0.1
        nn.init.constant_(self.linear1.bias, 0.1)
        nn.init.constant_(self.linear3.bias, 0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x


model = CartPoleDQN2()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
loss_fn = torch.nn.MSELoss()


def has_nan(model: nn.Module):
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False


for i in range(epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0
    # 随机门槛0.2开始
    threshold = 1 / ((i // 50) + 5)
    for j in range(500):
        # 根据模型选取动作
        old_state_tensor = torch.tensor(old_state)
        with torch.no_grad():
            action_distribution = model(old_state_tensor)
        action = torch.argmax(action_distribution).item()
        assert 0 <= action <= env.action_space.n
        # 如果小于门槛，就随机sample一个动作
        if random.random() < threshold:
            action = random.randint(0, env.action_space.n - 1)
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward
        # 新状态下, 根据当前策略最大概率的动作max_a对应的最大价值
        new_state_tensor = torch.tensor(new_state)
        with torch.no_grad():
            next_action_distribution = model(new_state_tensor.float())
        max_a = torch.max(next_action_distribution).item()

        # 反向传播
        model_output = model(old_state_tensor)
        if terminated or truncated:
            # 需要对齐的Q
            reward = -20
            ground_truth = reward + gama * max_a
            loss = loss_fn(model_output[action],
                           torch.tensor(ground_truth))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # 在反向传播后应用梯度裁剪
            optimizer.step()
        else:
            # 需要对齐的Q
            ground_truth = reward + gama * max_a
            loss = loss_fn(model_output[action],
                           torch.tensor(ground_truth))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # 在反向传播后应用梯度裁剪
            optimizer.step()

        if terminated or truncated:
            print("中止或者结束")
            print("loss: ", loss)
            print(list(model.parameters()))
            break

        old_state = new_state
    print("epoch: ", i)
    print("all_reward: ", all_reward)
    if has_nan(model):
        break
