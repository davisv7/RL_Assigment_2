import gym
from collections import defaultdict
from random import uniform, choice
import numpy as np
from math import exp


class DynaQ:

    def __init__(self, env):
        self.env = env

        self.episodes = 100000
        self.episode = 0

        self.learning_episodes = 1
        self.unplanned_episodes = 1000
        self.planning_steps = 10000

        self.test_cycles = 10

        self.alpha = 0.1
        self.gamma = 0.99
        self.action_space = list(range(self.env.action_space.n))
        self.state_space = np.array(range(16))

        self.q_dict = np.zeros((16, 4))
        self.t_dict = np.zeros((16, 4, 16))
        self.n_dict = np.zeros((16, 4, 16))
        self.r_dict = np.zeros((16, 4, 16))
        self.observed_states = defaultdict(set)

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
        for i in range(self.episodes):
            self.episode = i
            self.do_learning()
            _sum = 0
            for j in range(self.test_cycles):
                _sum += self.do_test()
            cum_reward += _sum
            if i % 1000 == 0:
                print("Win Rate after {} episodes: {}%".format(self.episode,
                                                               round(cum_reward / (i * self.test_cycles + 1), 4) * 100))

        print(cum_reward / (self.test_cycles * self.episodes))
        print(self.q_dict)

    def do_learning(self):
        """
        Learn in two stages:
            Q-learning, take actions in actual environment, update q value dictionary.
            Q-planning, take actions in modeled environment, update q value dictionary.
        :return:
        """
        for i in range(self.learning_episodes):
            # Reset will reset the environment to its initial configuration and return that state.
            current_state = self.env.reset()
            done = False

            while not done:
                # Q-learning
                action = self.epsilon_greedy(current_state)
                next_state, reward, done, _ = self.env.step(action)

                terminal = True if next_state == 15 else False
                self.q_update(current_state, next_state, action, reward, terminal)

                # Update n,r,t dictionaries
                self.n_dict[current_state][action][next_state] += 1
                self.r_dict[current_state][action][next_state] = reward
                total_transitions = np.sum(self.n_dict[current_state][action])
                self.t_dict[current_state][action] = np.divide(self.n_dict[current_state][action], total_transitions)

                # Q-Planning
                if i > self.unplanned_episodes:
                    self.do_planning()

                # add observed state and action, update current state
                if len(self.observed_states[current_state]) != 4:
                    self.observed_states[current_state].add(action)
                current_state = next_state

        self.env.close()

    def do_planning(self):
        """
        Use model to simulate taking actions at different states.
        Update q value dictionary accordingly.
        :return: None
        """
        if len(self.observed_states) > 0:
            for j in range(self.planning_steps):
                s = choice(list(self.observed_states.keys()))
                a = choice(tuple(self.observed_states[s]))
                s_prime = np.random.choice(self.state_space, 1, p=self.t_dict[s][a])  # <-- super slow
                r = self.r_dict[s][a][s_prime]
                done = True if s_prime == 15 else False
                self.q_update(s, s_prime, a, r, done)

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

    def q_update(self, s, s_prime, action, reward, done):
        """
        Update q dictionary.
        :param s: past state
        :param s_prime: resulting state
        :param action: action taken at past state to reach resulting state
        :param reward: reward at resulting state
        :return: None
        """
        if done:
            self.q_dict[s][action] += self.alpha * (reward - self.q_dict[s][action])
        else:
            old_val = self.q_dict[s][action]
            best_future_val = np.max(self.q_dict[s_prime])
            self.q_dict[s][action] = old_val + self.alpha * (reward + self.gamma * best_future_val - old_val)

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
        epsilon = exp((-self.episode - 400) / 4000) + 0.01
        return epsilon


def main():
    env = gym.make('FrozenLake-v0')

    q_obj = DynaQ(env)
    q_obj.do_run()


if __name__ == '__main__':
    main()
