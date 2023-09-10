'''
主要用来看一下LunarLander-v2的环境，没有写策略
'''

import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

result = []

for _ in range(1000):
    # agent policy that uses the observation and info
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    result.append([
        action, observation, reward, terminated, truncated, info
    ])

    if terminated or truncated:
        # observation, info = env.reset()
        break

result2 = [item[:3] for item in result]

observation, info = env.reset()
result3 = []

for _ in range(1000):
    # agent policy that uses the observation and info
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(3)

    result3.append([
        action, observation, reward, terminated, truncated, info
    ])

    if terminated or truncated:
        # observation, info = env.reset()
        break

env.close()
