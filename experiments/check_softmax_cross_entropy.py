import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

### Compute Softmax Cross Entropy

def log_softmax(x):
    x_max = np.max(x, axis=1).reshape(-1, 1)
    logsum = np.log(np.sum(np.exp(x - x_max), axis=1)).reshape(-1, 1)
    return x - x_max - logsum

def softmax_cross_entropy_with_logits(labels, logits):
    return -np.sum(labels * log_softmax(logits), axis=1)


X = 5 * np.random.randn(13, 7)
y = softmax(np.random.randn(13, 7))
res_tf = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=X).numpy()
res_np = softmax_cross_entropy_with_logits(labels=y, logits=X)
ts.check_tensors('softmax_cross_entropy_with_logits', res_tf, res_np)


### Compute Softmax Cross Entropy grad

# Start from the log-softmax grad operation:
# dX = dout - softmax(x) * sum(dout, axis=1).reshape(-1, 1)
#
# Combine with the rest of the cross-entropy loss to get the gradient
# t = - dL.reshape(-1, 1) * y
# dX = t - softmax(x) * sum(t, axis=1).reshape(-1, 1)
#    = -dL.reshape(-1, 1) * y + softmax(x) * sum(dL*y, axis=1)
# sum(y) = 1 => sum(DL*y) = Dl
# dX = softmax(x) * dL.reshape(-1, 1) - dL.reshape(-1, 1) * y
# dX = dL.reshape(-1, 1) * (softmax(x) - y)

def tf_softmax_cross_entropy_with_logits_grad(X, y, dout):
    vX = tf.Variable(X)
    with tf.GradientTape() as tape:
        out = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=vX)
        loss = tf.reduce_sum(out * dout)

    return tape.gradient(loss, vX).numpy()

def softmax_cross_entropy_with_logits_grad(X, y, dout):
    return (softmax(X) - y) * dout.reshape(-1, 1)

X = 5 * np.random.randn(13, 7)
y = softmax(np.random.randn(13, 7))
dout = np.random.randn(13)

res_tf = tf_softmax_cross_entropy_with_logits_grad(X, y, dout)
res_np = softmax_cross_entropy_with_logits_grad(X, y, dout)
ts.check_tensors('softmax_cross_entropy_with_logits_grad', res_tf, res_np)


ts.end()
