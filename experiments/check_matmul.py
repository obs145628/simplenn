import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()



### Forward pass

def matmul(lhs, rhs):
    return np.matmul(lhs, rhs)

X = np.random.randn(13, 7)
Y = np.random.randn(7, 29)

out_tf = tf.matmul(X, Y).numpy()
out_np = matmul(X, Y)
ts.check_tensors('matmul', out_tf, out_np)

### Backward pass


def tf_matmul_grad(lhs, rhs, dout):
    v_lhs = tf.Variable(lhs)
    v_rhs = tf.Variable(rhs)
    with tf.GradientTape() as tape:
        out = tf.matmul(v_lhs, v_rhs)
        loss = tf.reduce_sum(out * dout)

    d_lhs, d_rhs = tape.gradient(loss, [v_lhs, v_rhs])
    return d_lhs.numpy(), d_rhs.numpy()

def matmul_grad(lhs, rhs, dout):
    d_lhs = np.matmul(dout, rhs.T) 
    d_rhs = np.matmul(lhs.T, dout)
    return d_lhs, d_rhs

X = np.random.randn(13, 7)
Y = np.random.randn(7, 29)
dout = np.random.randn(13, 29)

dX_tf, dY_tf = tf_matmul_grad(X, Y, dout)
dX_np, dY_np = matmul_grad(X, Y, dout)
ts.check_tensors('matmul_grad_dX', dX_tf, dX_np)
ts.check_tensors('matmul_grad_dY', dY_tf, dY_np)


ts.end()
