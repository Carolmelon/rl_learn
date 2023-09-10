'''
主要用来看一下FrozenLake-v1的环境，没有写策略
'''

import gymnasium as gym
import time

# env = gym.make("FrozenLake-v1", render_mode="human")
env = gym.make('FrozenLake-v1',
               desc=None,
               map_name="4x4",
               is_slippery=False,   # 湖是不是滑的
               render_mode="human",
               )


observation, info = env.reset()
print(env.action_space)
print(env.observation_space)
result = []
for _ in range(100):
    # agent policy that uses the observation and info
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    result.append([
        action, observation, reward, terminated, truncated, info
    ])
    print(action)
    time.sleep(1)
    if terminated or truncated:
        # observation, info = env.reset()
        break

for i in range(4):
    observation, info = env.reset()
    print(env.action_space)
    print(env.observation_space)
    result = []
    for _ in range(10):
        # agent policy that uses the observation and info
        # action = env.action_space.sample()
        # 0 -> 左，1 -> 下，2 -> 右，3 -> 上
        action = i
        observation, reward, terminated, truncated, info = env.step(action)
        result.append([
            action, observation, reward, terminated, truncated, info
        ])
        print("action: ", action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("="*10)
        time.sleep(1)
        if terminated or truncated:
            print("terminated: ", terminated)
            print("truncated: ", truncated)
            print("="*20)
            # observation, info = env.reset()
            break
