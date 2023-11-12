import gym
import random
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env.reset()
# assigns rows(states) and columns(actions)
rows = env.observation_space.n
cols = env.action_space.n
qTable = np.zeros((rows, cols))
# how easily agent accepts new info
learningRate = 0.8
# how much long term rewards should be valued in comparison to short term
discountFactor = 0.8
epsilon = 0.85
decreasingRate = 0.005
episodes = 1000
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
            action = np.argmax(qTable[state, :])
        newState, reward, truncated, terminated, _ = env.step(action)
        qTable[state, action] = qTable[state, action] + learningRate * \
                                (reward + discountFactor * np.max(qTable[newState, :]) - qTable[state, action])
        print(f"Step: {step} of episode {episode}")
        state = newState
        if truncated or terminated:
            break
    epsilon = np.exp(-decreasingRate * episode)
    print(f"Epsilon: {epsilon}")

env = gym.make("FrozenLake-v1", render_mode="human")
episodes1 = 10
for episode1 in range(episodes1):
    score = 0
    step = 0
    state = env.reset()[0]
    truncated = False
    terminated = False
    while not(truncated or terminated):
        print(f"Step: {step}")
        action = np.argmax(qTable[state, :])
        newState, reward, truncated, terminated, _ = env.step(action)
        score += reward
        env.render()
        print(f"Score = {score}")
        state = newState
        step += 1
env.close()
