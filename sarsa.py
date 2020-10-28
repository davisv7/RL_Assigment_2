import gym
from random import uniform
import numpy as np
from math import exp, log
from results import Saver, Plotter


class Sarsa:

    def __init__(self, env):
        self.__name__ = "Sarsa"
        # Env Params
        self.env = env
        self.env.render()
        self.act_size = self.env.action_space.n
        self.action_space = list(range(self.act_size))
        self.obs_size = self.env.observation_space.n
        self.state_space = np.array(range(self.obs_size))

        # Learning Episode Params
        self.episodes = 100001
        self.episode = 1
        self.learning_episodes = 1
        self.test_cycles = 100
        self.testing_interval = 100

        # Learning Params
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = self.decay_function()
        self.q_dict = np.zeros((self.obs_size, self.act_size))

        self.filename = "{}_{}.csv".format(self.__name__, self.obs_size)
        self.saver = Saver(self.filename)

    def do_run(self):
        """
        For a number of episodes:
            do learning
            for a number of test cycles
                do testing
        Keep track of cumulative reward to be used for plotting.
        :return:
        """
        cum_reward = 0
        cum_avg = 0
        for i in range(1, self.episodes):
            self.episode = i
            self.do_learning()
            _sum = 0
            if i % self.testing_interval == 0:
                _sum = 0
                for j in range(self.test_cycles):
                    _sum += self.do_test()

                _avg = _sum / self.test_cycles
                cum_avg += _avg
                cum_reward += _sum
                test_rate = round(_avg * 100, 2)
                total_rate = round(cum_avg / (self.episode // self.testing_interval) * 100, 2)
                print("Win Rate after {} episodes: {}%".format(self.episode, test_rate))
                print("Total Win Rate: {}%\n".format(total_rate))
                self.saver.save([self.episode, test_rate, total_rate])
        self.saver.close()
        self.plotter = Plotter(self.filename)
        self.plotter.plot()


    def do_learning(self):
        """
        Learn in two stages:
            Q-learning, take actions in actual environment, update q value dictionary.
            Q-planning, take actions in modeled environment, update q value dictionary.
        :return:
        """
        for i in range(self.learning_episodes):

            # State, Action
            current_state = self.env.reset()
            old_action = self.epsilon_greedy(current_state)
            done = False

            while not done:
                # Reward, New State, New Action
                next_state, reward, done, _ = self.env.step(old_action)
                new_action = self.epsilon_greedy(next_state)

                # Q-Update
                terminal = True if next_state == self.obs_size - 1 else False
                self.q_update(current_state, next_state, old_action, new_action, reward, terminal)

                current_state = next_state
                old_action = new_action

        self.env.close()

    def do_test(self):
        """
        Run one-off test with current q value dictionary
        At each step, pick the action with the best q value.
        :return: reward (int)
        """
        # Reset will reset the environment to its initial configuration and return that state.
        current_state = self.env.reset()
        reward = 0
        done = False

        while not done:
            action = np.argmax(self.q_dict[current_state])
            next_state, reward, done, _ = self.env.step(action)
            current_state = next_state

        return reward

    def q_update(self, s, s_prime, old_action, new_action, reward, done):
        """
        Update q dictionary.
        :param s: past state
        :param s_prime: resulting state
        :param action: action taken at past state to reach resulting state
        :param reward: reward at resulting state
        :return: None
        """
        if done:
            self.q_dict[s][old_action] += self.alpha * (reward - self.q_dict[s][old_action])
        else:
            old_val = self.q_dict[s][old_action]
            new_val = self.q_dict[s_prime][new_action]
            self.q_dict[s][old_action] = old_val + self.alpha * (reward + self.gamma * new_val - old_val)

    def epsilon_greedy(self, state):
        """
        Generate random number and compare it to epsilon. If the number is less than epsilon, do random action.
        Else, do best action.
        :param state:
        :return: action (int)
        """
        chance = uniform(0, 1)
        self.epsilon = self.decay_function()
        if chance < self.epsilon:
            return self.env.action_space.sample()
        else:
            _max = np.max(self.q_dict[state])
            max_indices = np.argwhere(self.q_dict[state] == _max)
            if len(max_indices) > 1:
                return np.random.choice(np.concatenate(max_indices))
            else:
                return max_indices.item(0)

    def decay_function(self):
        # epsilon = min(log(2, exp(1)) / self.episode * 10000, 1)
        epsilon = exp((-self.episode - 400) / 4000) + 0.01
        return epsilon


def main():
    env_name = 'FrozenLake-v0'
    # map_name = '8x8'
    map_name = '4x4'
    env = gym.make(env_name, map_name=map_name)

    q_obj = Sarsa(env)
    q_obj.do_run()


if __name__ == '__main__':
    main()
