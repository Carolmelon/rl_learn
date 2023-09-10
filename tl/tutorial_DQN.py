"""
conda create --name tl python=3.8
conda activate tl
pip install tensorflow-gpu
pip install tensorlayer
pip install tensorflow-probability==0.9.0
pip install gym
pip install gym[atari]

python tutorial_DQN.py --train/test
"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

#####################  hyper parameters  ####################
env_id = 'FrozenLake-v1'
alg_name = 'DQN'
lambd = .99  # decay factor
e = 0.1  # e-Greedy Exploration, the larger the more random
num_episodes = 10000
render = True  # display the game environment

##################### DQN ##########################


def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


# Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(
        0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model):  # save trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


def load_ckpt(model):  # load trained weights
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


if __name__ == '__main__':

    qnetwork = get_model([None, 16])
    qnetwork.train()
    train_weights = qnetwork.trainable_weights

    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    env = gym.make(env_id, render_mode="human", is_slippery=False)

    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()[0]  # observation is state, integer 0 ~ 15
            rAll = 0
            if render:
                env.render()
            for j in range(99):  # step index, maximum step is 99
                # Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray(
                    [to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)

                # e-Greedy Exploration !!! sample random action
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                new_state, reward, terminated, truncated, info = env.step(a[0])
                s1, r, d, _ = new_state, reward, terminated or truncated, info
                if render:
                    env.render()
                # Obtain the Q' values by feeding the new state through our network
                Q1 = qnetwork(np.asarray(
                    [to_one_hot(s1, 16)], dtype=np.float32)).numpy()

                # Obtain maxQ' and set our target value for chosen action.
                # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + lambd * maxQ1
                # Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray(
                        [to_one_hot(s, 16)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(
                        targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                rAll += r
                s = s1
                # Reduce chance of random action if an episode is done.
                if d == True:
                    # reduce e, GLIE: Greey in the limit with infinite Exploration
                    e = 1. / ((i / 50) + 10)
                    break

            # Note that, the rewards here with random action
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'
                  .format(i, num_episodes, rAll, time.time() - t0))

            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(
                    all_episode_reward[-1] * 0.9 + rAll * 0.1)

        save_ckpt(qnetwork)  # save model
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        load_ckpt(qnetwork)  # load model
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            if render:
                env.render()
            for j in range(99):  # step index, maximum step is 99
                # Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray(
                    [to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                if render:
                    env.render()
                # Reduce chance of random action if an episode is done.
                if d:
                    break

            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'
                  .format(i, num_episodes, rAll, time.time() - t0))
