'''
策略梯度YYDS！
'''

import gymnasium as gym
import time
import numpy as np
import torch
from torch import nn
import random

env = gym.make('CartPole-v1', render_mode="human")

train_epoch = 100


class CartPolePG(nn.Module):
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=30, bias=True)
        self.linear3 = nn.Linear(
            in_features=30, out_features=out_features, bias=True)
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


class PG:
    def __init__(self) -> None:
        self.model = CartPolePG()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)
        self.actions = []
        self.states = []
        self.rewards = []
        self.gama = 0.9

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def add_record(self, action, state, reward):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)

    def choose_action_sample(self, state):
        '''
        state: 没有batch_size纬度
        '''
        with torch.no_grad():
            probs = self.model(state)
            probs = torch.nn.functional.softmax(probs, dim=-1)
        action = torch.multinomial(probs, num_samples=1)[0]
        return action.item()

    def choose_action_greedy(self, state):
        with torch.no_grad():
            probs = self.model(state)
        _, action = torch.max(probs, dim=-1)
        return action.item()

    def learn(self):
        discount_return = self.get_discount_return()
        # 先获取相应动作的分布
        self.states = torch.stack(self.states)  # [num_state, 4]
        action_dist = self.model(self.states)
        self.actions = torch.tensor(self.actions)
        # 不求和, 对制定的action求loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        # action_dist: [num_state, 2], self.actions: [num_state]
        inner_loss = criterion(action_dist, self.actions)
        loss = torch.mean(inner_loss * discount_return)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清除状态
        self.actions, self.states, self.rewards = [], [], []

    def get_discount_return(self, regularization=True):
        self.rewards = torch.tensor(self.rewards)
        discount_return = torch.zeros_like(self.rewards)
        tmp = 0
        for idx in reversed(range(len(self.rewards))):
            discount_return[idx] = self.gama * tmp + self.rewards[idx]
            tmp = discount_return[idx]
        if regularization:
            mean = discount_return.mean()
            std = discount_return.std()
            discount_return = (discount_return - mean) / std
        return discount_return

    def save_model(self, path=None):
        import os
        if path is None:
            # 获取当前脚本的文件路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对路径
            path = os.path.join(script_dir, 'save_model', 'cartpole_pg.pth')
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存模型
        torch.save(self.model, path)

    def load_model(self, path=None):
        import os
        if path is None:
            # 获取当前脚本的文件路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对路径
            path = os.path.join(script_dir, 'save_model', 'cartpole_pg.pth')
        # 加载模型
        self.model = torch.load(path)
        self.eval()


policyGradient = PG()
policyGradient.train()

for i in range(train_epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0

    for j in range(500):
        # 根据模型选取动作
        old_state_tensor = torch.tensor(old_state)
        with torch.no_grad():
            action = policyGradient.choose_action_sample(
                old_state_tensor)
        assert 0 <= action <= env.action_space.n
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward
        policyGradient.add_record(
            action=action, state=old_state_tensor, reward=reward
        )

        if terminated or truncated:
            policyGradient.learn()
            print("中止或者结束")
            print("i: ", i)
            print("all_reward: ", all_reward)
            print("="*30)
            break

        old_state = new_state

policyGradient.save_model()


policyGradient2 = PG()
policyGradient2.load_model()

test_epoch = 100

for i in range(test_epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0

    for j in range(500):
        # 根据模型选取动作
        old_state_tensor = torch.tensor(old_state)
        with torch.no_grad():
            action = policyGradient.choose_action_greedy(
                old_state_tensor)
        assert 0 <= action <= env.action_space.n
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward

        if terminated or truncated:
            print("测试中止或者结束")
            print("i: ", i)
            print("all_reward: ", all_reward)
            print("="*30)
            break

        old_state = new_state
