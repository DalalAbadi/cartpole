import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
import matplotlib
matplotlib.use("macOSX")
from matplotlib import pyplot as plt
import datetime
import tensorflow as tf
from keras.callbacks import TensorBoard
# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'rewards': []}
STATS_EVERY = 200
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# tensorboard_callback = TensorBoard(log_dir="logs")


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.0001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

            self.brain.fit(state, target_f, epochs=1, verbose=0,callbacks=[tensorboard_callback])

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)


    def run(self):
        try:
            for index_episode in range(self.episodes):
                episode_reward = 0
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
                ###############
                ep_rewards.append(episode_reward)
                #if not index_episode % STATS_EVERY:
                    #average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
                aggr_ep_rewards['ep'].append(index_episode)
                aggr_ep_rewards['rewards'].append(episode_reward)
                    #aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                    #aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
                    #print(
                        #f'Episode: {index_episode:>5d}, average reward: {average_reward:>4.1f},')

        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()




plt.scatter(aggr_ep_rewards['ep'], aggr_ep_rewards['rewards'], label="Rewards")

plt.legend(loc=4)
plt.show()