import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()

### Compute LogSoftmax

# More infos at https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better

X = 5 * np.random.randn(13, 7)

def log_softmax(x):
    x_max = np.max(x, axis=1).reshape(-1, 1)
    logsum = np.log(np.sum(np.exp(x - x_max), axis=1)).reshape(-1, 1)
    return x - x_max - logsum

res_tf = tf.nn.log_softmax(X, axis=1).numpy()
res_np = log_softmax(X)
ts.check_tensors('log_softmax', res_tf, res_np)


### Compute LogSoftmax grad

# More infos at https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
# gradInputi = gradOutputi - exp(outputi) . sum_j( gradOutputj )

def tf_log_softmax_grad(x, dy):
    vX = tf.Variable(X)
    with tf.GradientTape() as tape:
        y = tf.nn.log_softmax(vX, axis=1)
        loss = tf.reduce_sum(y * dy)

    return tape.gradient(loss, vX).numpy()

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

def log_softmax_grad(x, dy):
    return dy - softmax(x) * np.sum(dy, axis=1).reshape(-1, 1)

X = 5 * np.random.randn(13, 7)
dy = np.random.randn(13, 7)

res_tf = tf_log_softmax_grad(X, dy)
res_np = log_softmax_grad(X, dy)
ts.check_tensors('log_softmax_grad', res_tf, res_np)

ts.end()
