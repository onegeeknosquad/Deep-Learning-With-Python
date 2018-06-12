#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 08:58:33 2018

@author: mrpotatohead

Basic policy for CartPole-v0 that accelerates to the left if the pole is 
leaning to the left and accelerates right if the pole is leaning to the right.

Even with 500 tries, this policy never manages to keep the pole upright for 
more than 68 consecutive steps. Not great!
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

env = gym.make('CartPole-v0')
obs = env.reset()

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
    
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))