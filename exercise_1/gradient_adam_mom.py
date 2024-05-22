#!/usr/bin/env python
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER = 'output'
N_EPOCHS = 1000
OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, "exercise_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

experiments = {"Gradient_Descent": {"lr": [0.05, 0.1, 0.5, 1.0]},
               "Gradient_Descent-2": {"lr": [0.05, 0.005, 0.0005, 0.0001]},
               "Adam": {"lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]},
              "Momentum-0.2": {"lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], 
                           "momentum": 0.2},
              "Momentum-0.4": {"lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], 
                           "momentum": 0.4},
              "Momentum-0.6": {"lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], 
                           "momentum": 0.6},
              "Momentum-0.8": {"lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], 
                           "momentum": 0.8}}

for exercise_name, parameters in experiments.items():
  for LEARNING_RATE in parameters["lr"]:
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
    if "Gradient_Descent" in exercise_name:
      optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    elif "Adam" in exercise_name:
      optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    elif "Momentum" in exercise_name:
      optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, momentum=parameters["momentum"])

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

    plt.plot(range(N_EPOCHS), ls_loss, label=f"lr = {LEARNING_RATE}")
    # Use log scale to better visualize the loss
    plt.yscale("log")

  plt.legend()
  title = exercise_name + " with different LR "
  if "momentum" in parameters:
    title += " (Mom - " + str(parameters["momentum"]) + ")"
    
  plt.title(title)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  filename = exercise_name + "_lr_" + str(np.min(parameters["lr"])) + "_" + \
    str(np.max(parameters["lr"]))
  
  if "momentum" in parameters:
    filename += "_mom_" + str(parameters["momentum"])

  filename += ".png"

  plt.savefig(os.path.join(OUTPUT_DIR, filename), format="png", bbox_inches="tight")
  plt.clf()
