import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd

def chooseNextAction(Q, x, y, is_sarsa = True):
    epsilon = 0.1
    if np.random.random() < epsilon and is_sarsa:
        action = np.random.choice(range(4))
    else:
        max_value = -np.inf
        for action in range(4):
            value = Q[x,y,action]
            if max_value <= value:
                max_value = value
                max_action = action
        action = max_action
    return action

def chooseNextState(x, y, current_action):
    x_new, y_new = x, y
    if current_action == 0:
        x_new = x + 1
    elif current_action == 1:
        x_new = x - 1
    elif current_action == 2:
        y_new = y + 1
    elif current_action == 3:
        y_new = y - 1
    x_new = max(0, x_new)
    x_new = min(3, x_new)
    y_new = max(0, y_new)
    y_new = min(11, y_new)
    return x_new, y_new

def updateValueQ(Q, x, y, current_action, is_sarsa=False):
    alpha = 0.1
    gamma = 1
    x_new, y_new = chooseNextState(x, y, current_action)
    if x_new == 0 and y_new == 11:
        reward = 0
    elif x_new == 0 and 11 > y_new > 0:
        reward = -500
    else:
        reward = -1
    next_action = chooseNextAction(Q, x_new, y_new, is_sarsa)
    Q[x, y, current_action] = Q[x, y, current_action] + alpha * float((reward + gamma * Q[x_new, y_new, next_action] - Q[x, y, current_action]))
    if x_new == 0 and 11 > y_new > 0:
        return 0, 0, reward
    else:
        return x_new, y_new, reward

def algorithm(is_sarsa=False):
    actions_of_last_episode = [(0,0)]
    actions_per_episode = []
    rewards_per_episode = []
    Q = np.zeros((4, 12, 4))
    for n in range(1000):
        x = 0
        y = 0
        actions = 0
        rewards = 0
        while not(x == 0 and y == 11):
            current_action = chooseNextAction(Q, x, y)
            x_new, y_new, reward = updateValueQ(Q, x, y, current_action, is_sarsa=is_sarsa)
            x = x_new
            y = y_new
            actions = actions + 1
            rewards = rewards + reward
            if n == 999:
                actions_of_last_episode.append((x_new, y_new))
            if x == 0 and y == 0:
                actions_of_last_episode = [(0, 0)]
        actions_per_episode.append(actions)
        rewards_per_episode.append(rewards)
    
    return actions_of_last_episode, actions_per_episode, rewards_per_episode
    


q_steps, q_actions, q_rewards = algorithm()
print "Q-Learning Steps"
for step in q_steps:
    print step

sarsa_steps, sarsa_actions, sarsa_rewards = algorithm(is_sarsa = True)
print "Sarsa-Learning Steps"
for step in sarsa_steps:
    print step

plot(q_steps, q_actions, q_rewards, sarsa_steps, sarsa_actions, sarsa_rewards)
