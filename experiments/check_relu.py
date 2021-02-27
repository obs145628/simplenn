import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()

### Forward pass

def relu(x):
    return np.maximum(x, 0)

x = np.random.randn(7, 9, 3)
res_tf = tf.nn.relu(x).numpy()
res_np = relu(x)
ts.check_tensors('relu', res_tf, res_np)


### Backward pass

def tf_relu_grad(x, dout):
    v_x = tf.Variable(x)
    with tf.GradientTape() as tape:
        out = tf.nn.relu(v_x)
        loss = tf.reduce_sum(out * dout)

    return tape.gradient(loss, v_x).numpy()

def relu_prime(x):
    return np.where(x > 0, 1.0, 0.0)

def relu_grad(x, dout):
    return dout * relu_prime(x)

x = 5 * np.random.randn(7, 9, 3)
dout = 5 * np.random.randn(7, 9, 3)

res_tf = tf_relu_grad(x, dout)
res_np = relu_grad(x, dout)
ts.check_tensors('relu_grad', res_tf, res_np)

ts.end()
