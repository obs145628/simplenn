import os
import sys
import numpy as np
import tensorflow as tf

from nn.graph import Graph
import nn.model as model

from datasets import get_mnist
import nn_model
import tf_model
from barray import BArrayOs

from tester import Tester

BATCH_SIZE=32
LEARNING_RATE=1e-3
NUM_EPOCHS = 25


(X_train, y_train), (X_test, y_test) = get_mnist()
print(X_train.shape)
print(y_train.shape)

model_base, model_probs, optimizer = nn_model.build_nn_model(BATCH_SIZE,
                                                             LEARNING_RATE)

tf_model_base, tf_model_probs, tf_weights = tf_model.build_tf_model()

def check_forward():
    model_base.assign_weights(tf_weights)
    tester = Tester()

    
    tf_logits = tf_model_base(X_train[:BATCH_SIZE]).numpy()
    nn_logits = model_base(X_train[:BATCH_SIZE], dump_code=True)
    tester.check_tensors('logits', tf_logits, nn_logits)

    tf_probs = tf_model_probs(X_train[:BATCH_SIZE]).numpy()
    nn_probs = model_probs(X_train[:BATCH_SIZE])
    tester.check_tensors('probs', tf_probs, nn_probs)

    tf_loss = tf_model.compute_loss(tf_model_base,
                                    labels=y_train[:BATCH_SIZE],
                                    input=X_train[:BATCH_SIZE]).numpy()
    nn_loss = optimizer.loss(labels=y_train[:BATCH_SIZE], input=X_train[:BATCH_SIZE])
    tester.check_tensors('loss', tf_loss, nn_loss)

    
    tester.end()

def check_backward():
    model_base.assign_weights(tf_weights)
    tester = Tester()

    optimizer.dump_gradients_code()
    
    
    tf_grads = tf_model.compute_gradients(tf_model_base,
                                          labels=y_train[:BATCH_SIZE],
                                          input=X_train[:BATCH_SIZE])
    np_grads = optimizer.compute_gradients(labels=y_train[:BATCH_SIZE],
                                           input=X_train[:BATCH_SIZE])
    for (idx, (tf_grad, np_grad)) in enumerate(zip(tf_grads, np_grads)):
        tester.check_tensors('grad_{}'.format(idx), tf_grad.numpy(), np_grad)

    
    tester.end()

def train():
    optimizer.train(X_train, y_train, X_test, y_test,
                    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                    dump_code=True)

def export_data():
    export_dir = './export'
    os.makedirs(export_dir, exist_ok=True)
    mnist_train_file = os.path.join(export_dir, 'mnist-train.bin')
    mnist_test_file = os.path.join(export_dir, 'mnist-test.bin')
    weigths_file = os.path.join(export_dir, 'weights.bin')
    results_file = os.path.join(export_dir, 'results.bin')
    grads_file = os.path.join(export_dir, 'grads.bin')

    tf_grads = tf_model.compute_gradients(tf_model_base,
                                          labels=y_train[:BATCH_SIZE],
                                          input=X_train[:BATCH_SIZE])
    tf_grads = [t.numpy() for t in tf_grads]

    tf_logits = tf_model_base(X_train[:BATCH_SIZE]).numpy()
    tf_probs = tf_model_probs(X_train[:BATCH_SIZE]).numpy()
    tf_loss = tf_model.compute_loss(tf_model_base,
                                    labels=y_train[:BATCH_SIZE],
                                    input=X_train[:BATCH_SIZE]).numpy()

    
    BArrayOs.write_tensors_to_file(mnist_train_file, [X_train, y_train])
    BArrayOs.write_tensors_to_file(mnist_test_file, [X_test, y_test])
    BArrayOs.write_tensors_to_file(weigths_file, tf_weights)
    BArrayOs.write_tensors_to_file(results_file, [tf_logits, tf_probs, tf_loss])
    BArrayOs.write_tensors_to_file(grads_file, tf_grads)
    

if __name__ == '__main__':
    args = list(sys.argv)

    if '--forward' in args:
        check_forward()
    elif '--backward' in args:
        check_backward()
    elif '--train' in args:
        train()
    elif '--export' in args:
        export_data()

    else:
        print('Unknown option')
        print('Usage:')
        print('    --foward: Run tests for forward pass')
        print('  --backward: Run tests for backward pass')
        print('     --train: Run training with SGD')
        print('    --export: Export all data for future use')
        sys.exit(1)
