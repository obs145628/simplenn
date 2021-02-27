import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()



### Forward pass

def sum(x, axis):
    return np.sum(x, axis=axis)

X = np.random.randn(456)
res_tf = tf.reduce_sum(X, axis=0).numpy()
res_np = sum(X, axis=0)
ts.check_tensors('sum1', res_tf, res_np)

### Backward pass

def tf_sum_grad(x, axis, dout):
    vX = tf.Variable(X)
    with tf.GradientTape() as tape:
        out = tf.reduce_sum(vX, axis=axis)
        loss = tf.reduce_sum(out * dout)

    return tape.gradient(loss, vX).numpy()

def sum_grad(x, axis, dout):
    shape = list(x.shape)
    shape[axis] = 1
    multiplier = [1] * len(x.shape)
    multiplier[axis] = x.shape[axis]
    
    dout = np.reshape(dout, shape)
    return np.tile(dout, multiplier)

X = np.random.randn(456)
dout = np.random.randn()
res_tf = tf_sum_grad(X, axis=0, dout=dout)
res_np = sum_grad(X, axis=0, dout=dout)
ts.check_tensors('sum_grad1', res_tf, res_np)


X = np.random.randn(3, 4, 5)
dout = np.random.randn(4, 5)
res_tf = tf_sum_grad(X, axis=0, dout=dout)
res_np = sum_grad(X, axis=0, dout=dout)
ts.check_tensors('sum_grad2', res_tf, res_np)

X = np.random.randn(3, 4, 5)
dout = np.random.randn(3, 5)
res_tf = tf_sum_grad(X, axis=1, dout=dout)
res_np = sum_grad(X, axis=1, dout=dout)
ts.check_tensors('sum_grad3', res_tf, res_np)

X = np.random.randn(3, 4, 5)
dout = np.random.randn(3, 4)
res_tf = tf_sum_grad(X, axis=2, dout=dout)
res_np = sum_grad(X, axis=2, dout=dout)
ts.check_tensors('sum_grad4', res_tf, res_np)

ts.end()
