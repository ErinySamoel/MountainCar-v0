import numpy as np
import gym
import pickle
import gym
import random
import pygame
from pygame import gfxdraw
env = gym.make('MountainCar-v0')

# Discritize observation and action space in bins.
pos_space = np.linspace(-1.2, 0.6, 18)
vel_space = np.linspace(-0.07, 0.07, 28)
# given observation, returns what bin
def getState(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)

    return (pos_bin, vel_bin)


# Creates a new empty Q-table for this environment
def createEmptyQTable():
    states = []
    for pos in range(len(pos_space) + 1):
        for vel in range(len(vel_space) + 1):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in range(env.action_space.n):
            Q[state, action] = 0
    return Q


# Given a state and a set of actions
# returns action that has the highest Q-value
def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action

# Hyperparameters
Q = createEmptyQTable()
alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Factor
epsilon = 1  # e-Greedy
episodes = 5000  # number of episodes
score = 0
# Variable to keep track of the total score obtained at each episode
total_score = np.zeros(episodes)

for i in range(episodes):
    if i % 500 == 0:
        print(f'episode: {i}, score: {score}, epsilon: {epsilon:0.3f}')

    observation = env.reset()
    state = getState(observation)

    # e-Greedy strategy
    # Explore random action with probability epsilon
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    # Take best action with probability 1-epsilon
    else:
        action = maxAction(Q, state)

    score = 0
    done = False
    while not done:
        # Take action and observe next state
        next_observation, reward, done, info = env.step(action)
        next_state = getState(next_observation)

        # Get next action following e-Greedy policy
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = maxAction(Q, next_state)

        # Add reward to the score of the episode
        score += reward

        # Update Q value for state and action given the bellman equation
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        # Move to next state, and next action
        state, action = next_state, next_action

    total_score[i] = score
    epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01
# Run 10 episodes
import time
start=time.time()
for episode in range(10):
    done = False
    observation = env.reset()
    state = getState(observation)
    # While the car don't reach the goal or number of steps < 200
    while not done:
        env.render()
        print(observation)
        # Take the best action for that state given trained values
        action = maxAction(Q, state)
        observation, reward, done, info = env.step(action)
        # Go to next state
        state = getState(observation)
time.sleep(1)
end = time.time()
print(end -start)
#acc = score/episodes
acc=total_score/episodes
acc=np.mean(abs(acc))
print(acc)
env.close()
import pickle

# save
with open('SARSA.pkl','wb') as f:
    pickle.dump(Q,f)

# load
with open('SARSA.pkl', 'rb') as f:
    t = pickle.load(f)
