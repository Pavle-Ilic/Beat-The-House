from __future__ import annotations
import gym
import random
import numpy as np
from collections import defaultdict

# returns taxis library (states and actions)
env = gym.make("Blackjack-v1", render_mode="rgb_array")
# resets environment
env.reset()
qValues = defaultdict(lambda: np.zeros(env.action_space.n))
# how easily agent accepts new info
learningRate = 0.01
# how much long term rewards should be valued in comparison to short term
discountFactor = 0.95
epsilon = 1
episodes = 100000
# decreasing rate of epsilon
decreasingRate = epsilon / (episodes / 2)
steps = 100

for episode in range(episodes):
    truncated = False
    terminated = False
    state = env.reset()[0]
    for step in range(steps):
        # determines whether agent explores or exploits
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qValues[state])
            print(f"action: {action}")
        newState, reward, truncated, terminated, _ = env.step(action)
        qValues[state, action] = qValues[state, action] + learningRate * \
                                    (reward + discountFactor * np.max(qValues[newState]) - qValues[state, action])
        print(f"episode: {episode}")
        state = newState
        if truncated or terminated:
            break
    epsilon = np.exp(-decreasingRate * episode)
    # print(f"Epsilon: {epsilon}")

env = gym.make("Blackjack-v1", render_mode="human")
episodes1 = 10
for episode1 in range(episodes1):
    score = 0
    state = env.reset()[0]
    truncated = False
    terminated = False
    while not(truncated or terminated):
        action = np.argmax(qValues[state])
        if action == 1:
            print(f"action = hit")
        newState, reward, truncated, terminated, _ = env.step(action)
        score += reward
        env.render()
        print(f"Score = {score}")
        state = newState
env.close()
