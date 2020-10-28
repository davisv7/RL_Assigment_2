# A sample agent that just takes random actions in whatever environment you choose. I've included the 4x4 version and 8x8 versions of Frozen Lake for you to play around with. I've also put CartPole in there because I think it's neat. If you try to activate cartpole, disable the action and observation space print statements.

import gym
import random
from time import sleep
# Create the environment
env = gym.make('FrozenLake8x8-v0')
# env = gym.make('FrozenLake-v0')
# env= gym.make('CartPole-v0')

# Print out the number of actions and states in the environment (disable for cartpole)
print(env.action_space.n)
print(env.observation_space.n)

# The number of episodes the agent will explore
numEps = 1

for i in range(numEps):
    # Reset will reset the environment to its initial configuration and return that state.
    currentState = env.reset()

    done = False
    stepCount = 0

    # Loop until either the agent finishes or takes 200 actions, whichever comes first.
    while stepCount < 200 and done == False:
        stepCount += 1
        # Sample a random action from the action_space
        actionToTake = env.action_space.sample()
        print("Action Taken: " + str(actionToTake))
        # Execute actions using the step function.
        # Returns the nextState, reward, a boolean indicating whether this is a terminal state.
        # The final thing it returns is a probability associated with the underlying transition distribution,
        # but we shouldn't need that for this assignment.
        nextState, reward, done, _ = env.step(actionToTake)

        # Render visualizes the environment
        env.render()
        currentState = nextState
env.close()
