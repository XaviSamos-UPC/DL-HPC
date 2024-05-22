#!/usr/bin/env python
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER = 'output'

N_EPOCHS = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.8
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, "exercise_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ls_optimizer = [tf.train.GradientDescentOptimizer(LEARNING_RATE),
                 tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM),
                 tf.train.AdamOptimizer(0.05)]

for optimizer in ls_optimizer:
  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)

  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  train = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


  ls_loss = []
  for i in range(N_EPOCHS):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    ls_loss.append(curr_loss)

  plt.plot(range(N_EPOCHS), ls_loss, label=f"{optimizer.__class__.__name__}")
  # Use log scale to better visualize the loss
  plt.yscale("log")

plt.legend()
plt.title(f"Different Optimizers with LR = {LEARNING_RATE}")
plt.xlabel("Epochs")
plt.ylabel("Loss")

filename = f"Different_Optimizers_lr_{LEARNING_RATE}".replace(".", "_") + ".png"

plt.savefig(os.path.join(OUTPUT_DIR, filename), format="png", bbox_inches="tight")
