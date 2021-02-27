import numpy as np
import tensorflow as tf

def to_one_hot(data, num_classes):
    res = np.zeros([len(data), num_classes]).astype(np.float32)
    for (i, x) in enumerate(data):
        res[i][x] = 1
    return res

def get_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)
    y_train = to_one_hot(y_train, num_classes=10)
    y_test = to_one_hot(y_test, num_classes=10)
    return (X_train, y_train), (X_test, y_test)
