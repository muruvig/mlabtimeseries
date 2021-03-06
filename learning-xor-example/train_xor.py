# Arda Sahiner
# Attempting to train an RNN composed of LSTM cells to learn XOR
# Topology:
#   Two LSTM cells, the first which takes one bit, and the second which takes the other
#   Output is read out of the second LSTM cell
# As of right now, a lot of this doesn't work
# Inspo from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import math

# filename_queue = tf.train.string_input_producer(["xor.csv"])
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# Parameters
learning_rate = 0.01
training_iters = 1000
display_step = 1
batch_size = 10

# test_iters = 1000
# Network Parameters
n_input = 1 #scalar inputs at each time step
n_steps = 2 # timesteps
n_hidden = 3 # hidden layer num of features
n_classes = 2 #either 0 or 1


def dense_to_one_hot(labels_dense, num_classes = n_classes):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  record_defaults = [[1], [1], [1]]
  col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
  features = tf.pack([col1, col2])
  return features, col3

def input_pipeline(batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      ["xor.csv"], shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch.eval(), dense_to_one_hot(label_batch.eval())

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


with tf.Session() as sess:

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.initialize_all_variables()

    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        print(step)
        batch_x, batch_y = input_pipeline(batch_size)
        print("got batches")

        bx = batch_x
        print("evaled x")

        by = batch_y
        print("evaled y")

        # bx, by = sess.run([batch_x, batch_y])
        # print("done w sess run first")
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: bx, y: by})
        print("done w optimizer")
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: bx, y: by})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: bx, y: by})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    #@TODO: test


    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")
