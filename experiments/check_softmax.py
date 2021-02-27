import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tester import Tester

ts = Tester()

X = 5 * np.random.randn(12, 6)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

res_tf = tf.nn.softmax(X, axis=1).numpy()
res_np = softmax(X)
ts.check_tensors('softmax', res_tf, res_np)

ts.end()
