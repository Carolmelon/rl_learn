import gymnasium as gym
import time
import numpy as np

# env = gym.make("FrozenLake-v1", render_mode="human")
env = gym.make('FrozenLake-v1',
               desc=None,
               map_name="4x4",
               is_slippery=False,   # 湖是不是滑的
               render_mode="human",
               )

Q = np.zeros([env.observation_space.n, env.action_space.n])
epoch = 500
alpha = 0.85
gama = 0.99


def get_row_col(idx, cols=4):
    return (idx // cols + 1, idx % cols + 1)


for i in range(epoch):
    old_state, info = env.reset()   # s:状态
    all_reward = 0
    for j in range(99):
        # 选取动作
        action_distribution = Q[old_state, :] + \
            np.random.randn(1, env.action_space.n) * 1/(i+1)
        action = np.argmax(action_distribution)
        new_state, reward, terminated, truncated, info = env.step(action)

        all_reward += reward
        # 新状态下, 根据当前策略最大概率的动作max_a
        max_a = np.argmax(Q[new_state, :])
        Q[old_state, action] = Q[old_state, action] + \
            alpha*(reward + gama * Q[new_state, max_a] - Q[old_state, action])
        if terminated or truncated:
            print("old_state: ", get_row_col(old_state))
            print("new_state: ", get_row_col(new_state))
            break
        old_state = new_state
    if all_reward != 0:
        print(Q)
    print("epoch: ", i)
    print("all_reward: ", all_reward)

print(Q)
