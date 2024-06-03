#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import read_inputs
import numpy as N
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_FOLDER = "output"
EXERCISE = "exercise_2"

N_EPOCHS = 30
ls_lr = [0.00001, 0.0001, 0.001, 0.005]

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

OUTPUT_DIR = os.path.join(OUTPUT_FOLDER, EXERCISE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
data = data_input[0]

# Data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float32 )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

# Data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float32 )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0

# Set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

for exercise_name, parameters in experiments.items():

  for LEARNING_RATE in parameters["lr"]:

    with tf.device('/gpu:0'):
      train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

      sess = tf.InteractiveSession()
      tf.global_variables_initializer().run()

      # TRAINING PHASE
      print("TRAINING")

      ls_loss = []
      ls_acc = []
      for epoch in range(N_EPOCHS):
        for i in range(500):
          batch_xs = data[0][0][100*i:100*i+100]
          batch_ys = real_output[100*i:100*i+100]
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # CALCULATING THE LOSS
        loss = sess.run(cross_entropy, feed_dict={x: data[0][0], y_: real_output})
        
        #CHECKING THE ERROR
        # print("ERROR CHECK")

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ACC = sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check})
        print(f"EPOCH {epoch}({N_EPOCHS}) -- LR: {LEARNING_RATE} -- Loss: {loss} --- ACC: {ACC}")

        ls_loss.append(loss)
      ls_acc.append(ACC)
    
    plt.plot(range(N_EPOCHS), ls_loss, label=f"lr = {LEARNING_RATE}")
    # Use log scale to better visualize the loss
    plt.yscale("log")

  title = exercise_name + " with different LR "
  if "momentum" in parameters:
    title += " (Mom - " + str(parameters["momentum"]) + ")"

  plt.legend()
  plt.title(title)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  filename = exercise_name + "_lr_" + str(np.min(parameters["lr"])) + "_" + \
    str(np.max(parameters["lr"]))
  
  if "momentum" in parameters:
    filename += "_mom_" + str(parameters["momentum"])

  filename += ".png"
  plt.savefig(os.path.join(OUTPUT_DIR, filename), format="pdf", bbox_inches="tight")
  plt.clf()

