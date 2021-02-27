import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()



### Forward pass

def bias_add(lhs, rhs):
    return np.add(lhs, rhs)

X = np.random.randn(13, 7)
y = np.random.randn(7)

out_tf = tf.nn.bias_add(X, y).numpy()
out_np = bias_add(X, y)
ts.check_tensors('bias_add', out_tf, out_np)

### Backward pass


def tf_bias_add_grad(lhs, rhs, dout):
    v_lhs = tf.Variable(lhs)
    v_rhs = tf.Variable(rhs)
    with tf.GradientTape() as tape:
        out = tf.nn.bias_add(v_lhs, v_rhs)
        loss = tf.reduce_sum(out * dout)

    d_lhs, d_rhs = tape.gradient(loss, [v_lhs, v_rhs])
    return d_lhs.numpy(), d_rhs.numpy()

def bias_add_grad(lhs, rhs, dout):
    d_lhs = dout
    d_rhs = np.sum(dout, axis=0)
    return d_lhs, d_rhs

X = np.random.randn(13, 7)
y = np.random.randn(7)
dout = np.random.randn(13, 7)

dX_tf, dy_tf = tf_bias_add_grad(X, y, dout)
dX_np, dy_np = bias_add_grad(X, y, dout)
ts.check_tensors('bias_add_grad_dX', dX_tf, dX_np)
ts.check_tensors('bias_add_grad_dy', dy_tf, dy_np)


ts.end()
