'''
改编自：https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_DQN.py

暂时不太行

还不能运行，原因是使用了老版本的gym

'''


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义DQN模型


class CartPoleDQN(nn.Module):
    def __init__(self):
        super(CartPoleDQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化环境和模型
env = gym.make('CartPole-v1')
model = CartPoleDQN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
num_episodes = 1000
gamma = 0.99

# 开始训练
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        q_values = model(state)
        action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # 计算下一个状态的最大Q值
        with torch.no_grad():
            next_q_values = model(next_state)
            max_q_value = torch.max(next_q_values).item()

        # 计算目标值
        target = reward + gamma * max_q_value

        # 计算损失
        loss = loss_fn(q_values[action], torch.tensor(target))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 使用训练好的模型进行测试
state = env.reset()
done = False
total_reward = 0

while not done:
    state = torch.tensor(state, dtype=torch.float32)
    q_values = model(state)
    action = torch.argmax(q_values).item()

    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Test Total Reward: {total_reward}")
