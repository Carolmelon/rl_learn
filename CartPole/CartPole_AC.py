'''
试一下用actor-critic方法写CartPole

110次尝试以后能收敛了
'''

import os
import datetime
import gymnasium as gym
import time
import numpy as np
import torch
from torch import nn
import random
import torch.nn.utils as torch_utils

from torch.utils.tensorboard import SummaryWriter

# 获取当前代码文件名（不包含路径）
current_file = os.path.basename(__file__)

# 构建日志文件名
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'./logs/{current_file}_{current_time}'
# 创建SummaryWriter对象并指定日志文件名和保存路径
writer = SummaryWriter(log_dir=log_dir)

env = gym.make('CartPole-v1', render_mode="human")


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


class CartPoleActorModel(nn.Module):
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        inner_dim1 = 30
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=inner_dim1, bias=True)
        self.linear3 = nn.Linear(
            in_features=inner_dim1, out_features=out_features, bias=True)
        self.act_fn = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 初始化权重w为均值为0，方差为0.03
                nn.init.normal_(module.weight, mean=0, std=0.03)
                # 初始化偏置b为常数0.1
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x


class CartPoleCriticModel(nn.Module):
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        inner_dim1 = 30
        inner_dim2 = 20
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=inner_dim1, bias=True
        )
        # self.linear2 = nn.Linear(
        #     in_features=inner_dim1, out_features=inner_dim2, bias=True
        # )
        self.linear3 = nn.Linear(
            in_features=inner_dim1, out_features=out_features, bias=True
        )
        self.act_fn = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 初始化权重w为均值为0，方差为0.03
                nn.init.normal_(module.weight, mean=0, std=0.03)
                # 初始化偏置b为常数0.1
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.act_fn(x)
        # x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x


class CartPoleAC:
    def __init__(self) -> None:
        # 输出2维向量，表示动作概率分布 A_t = Pi(S_t)
        self.actor = CartPoleActorModel(in_features=4, out_features=2)
        # 输出一个标量值，表示 V(S_t)
        self.critic = CartPoleCriticModel(in_features=4, out_features=1)
        self.model = torch.nn.ModuleList([self.actor, self.critic])
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-2)
        self.gama = 0.9
        self.step = 0
        self.epoch = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def choose_action_sample(self, state):
        '''
        state: 没有batch_size纬度
        '''
        with torch.no_grad():
            probs = self.actor(state)
            probs = torch.nn.functional.softmax(probs, dim=-1)
        action = torch.multinomial(probs, num_samples=1)[0]
        return action.item()

    def choose_action_greedy(self, state):
        with torch.no_grad():
            probs = self.actor(state)
        _, action = torch.max(probs, dim=-1)
        return action.item()

    def change_learn_order(self):
        # if self.epoch < 20:
        #     self.learn_order = ['critic',]
        #     return
        self.learn_order = ['critic', 'actor']
        return

        order = self.epoch // 10
        if order % 2 == 0:
            self.learn_order = ['critic',]
        else:
            self.learn_order = ['actor',]

    def learn_ac(self, old_state, new_state, action, reward):
        if self.start_epoch == True:
            self.change_learn_order()
            self.start_epoch = False

        # if self.step % 100 == 1:
        #     print("step: {}".format(self.step))
        # self.step += 1

        if 'critic' in self.learn_order:
            # 先学critic吧
            old_state_value = self.critic(old_state)
            with torch.no_grad():
                new_state_value = self.critic(new_state)
            critic_criterion = nn.MSELoss()
            td_target = reward + self.gama * new_state_value
            critic_loss = 1/2 * critic_criterion(old_state_value, td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # torch_utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

        if 'actor' in self.learn_order:
            # 学actor
            with torch.no_grad():
                old_state_value = self.critic(old_state)
                new_state_value = self.critic(new_state)
            td_target = reward + self.gama * new_state_value
            td_error = td_target.detach().clone() - old_state_value.detach().clone()
            if torch.abs(td_error) > 5:
                pass
            action_dist_on_old_state = self.actor(old_state)
            actor_criterion = nn.CrossEntropyLoss()
            loss1 = actor_criterion(
                action_dist_on_old_state,
                action,
            )
            loss2 = loss1 * td_error
            self.actor_optimizer.zero_grad()
            loss2.backward()
            torch_utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

    def save_model(self, path=None):
        import os
        if path is None:
            # 获取当前脚本的文件路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对路径
            path = os.path.join(script_dir, 'save_model', 'cartpole_ac.pth')
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
            path = os.path.join(script_dir, 'save_model', 'cartpole_ac.pth')
        # 加载模型
        self.model = torch.load(path)
        self.eval()


actorCritic = CartPoleAC()
actorCritic.train()

train_epoch = 500

for i in range(train_epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0

    actorCritic.start_epoch = True

    for j in range(500):
        # 根据模型选取动作
        old_state_tensor = torch.tensor(old_state)
        action = actorCritic.choose_action_sample(old_state_tensor)
        assert 0 <= action <= env.action_space.n
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward

        if terminated or truncated:
            reward = -20

        actorCritic.learn_ac(
            old_state=torch.tensor(old_state),
            new_state=torch.tensor(new_state),
            action=torch.tensor(action),
            reward=torch.tensor(reward),
        )

        def tab_print(x):
            print("\t" + x)

        if terminated or truncated:
            writer.add_scalar('Loss/train', all_reward, i)
            if terminated:
                tab_print("中止")
            if truncated:
                tab_print("结束")
            tab_print("epoch: {}".format(i))
            tab_print("all_reward: {}".format(all_reward))
            print("="*30)
            break

        old_state = new_state

    actorCritic.epoch += 1

writer.close()
actorCritic.save_model()


actorCritic2 = CartPoleAC()
actorCritic2.load_model()

test_epoch = 100

exit(0)

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
