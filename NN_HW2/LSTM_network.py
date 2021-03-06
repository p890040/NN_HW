import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 6
display_step = 500

# Network Parameters
n_input = 2 # MNIST data input (img shape: 28*28)
n_steps = 8 # timesteps
n_hidden = 300 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


x_data = np.loadtxt('in_train.txt')/100
y_data = np.loadtxt('out_train.txt')[: , np.newaxis]
x_data_test = np.loadtxt('in_test.txt')/100
y_data_test = np.loadtxt('out_test.txt')[: , np.newaxis]
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


#batch_data = next_batch(train_data, batch_size)
#batch_target = next_batch(train_target, batch_size)
#feed = {data: batch_data, target: batch_target}

curr=0
def next_batch_data(dataset, batch_size):
    global curr
    size = dataset.shape[0]
    if(curr + batch_size > size):
        B = np.concatenate((dataset[curr:size], dataset[0:batch_size-(size-curr)]))
        curr = batch_size-(size-curr)
        return B
    else:
        B = dataset[curr:curr+batch_size]
        curr = curr+batch_size
        return B

curr2=0
def next_batch_target(dataset, batch_size):
    global curr2
    size = dataset.shape[0]
    if(curr2 + batch_size > size):
        B = np.concatenate((dataset[curr2:size], dataset[0:batch_size-(size-curr2)]))
        curr2 = batch_size-(size-curr2)
        return B
    else:
        B = dataset[curr2:curr2+batch_size]
        curr2 = curr2+batch_size
        return B


def hot_code(y):
    length = len(y)
    t = np.zeros((length,10))
    for i in range(length):
        t[i][(int)(y[i])] = 1
    return t

y_data_hotcode = hot_code(y_data)
y_data_test_hotcode = hot_code(y_data_test)
        
    


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    prediction = []
    for i in range(len(outputs)):
        prediction.append(tf.matmul(outputs[i], weights['out']) + biases['out'])
    return prediction

pred = RNN(x, weights, biases)[-1]

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        
        batch_data = next_batch_data(x_data, batch_size)
        batch_data = batch_data.reshape((batch_size, n_steps, n_input))
        
        batch_target = next_batch_target(y_data_hotcode, batch_size)
        feed = {x: batch_data, y: batch_target}
        # Reshape data to get 28 seq of 28 elements
        # Run optimization op (backprop)
        sess.run(optimizer, feed)
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_data, y: batch_target})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_data, y: batch_target})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = len(x_data_test)
    test_data = x_data_test[:test_len].reshape((-1, n_steps, n_input))
    test_label = y_data_test_hotcode[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
