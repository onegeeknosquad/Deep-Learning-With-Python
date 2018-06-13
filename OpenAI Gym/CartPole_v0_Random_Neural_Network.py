#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:17:07 2018

@author: mrpotatohead
"""

import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
obs = env.reset()

n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.variance_scaling_initializer()

#Build the Neural Network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

#Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right),num_samples=1)

init = tf.global_variables_initializer()



n_max_steps = 1000000
frames = []

with tf.Session() as sess:
    init.run()
    obs = env.reset()
    for step in range(n_max_steps):
        action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
        env.render()
        if done:
            print("PENIS")
            break
        
env.close()