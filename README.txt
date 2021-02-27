# Presentation

Really simple implementation of a neural network with dense layer, relu activations and softmax cross-entropy loss.
The full implementation is done only with numpy
It uses a computation graph and automatic differentation
SGD Optimizer
A sample network is trained on the MNIST dataset

Tensorflow is only used as a reference to check the correctness of the computations


# Prepare env

python3 -m venv _env
. _env/bin/activate
pip install --upgrade pip
pip install tensorflow numpy

# Run Basic experiments

python experiments/check_<xxx>.py


# Run main program

## Check forward pass

python main.py --forward

## Check backward pass

python main.py --backward

## Train the network

python main.py --train

## Export data

Export data for use by non python code
create folder export at root with:
- mnist-train.bin: training datset X_train and y_train for mnist
- mnist-test.bin: testing datset X_test and y_test for mnist
- weights.bin: tensorflow weights value
- results.bin: corresponding model output given these weight and X_train[0:BATCH_SIZE], y_train[0:BATCH_SIZE] used
- grads.bin: corresponding gradients value given these weights and X_train[0:BATCH_SIZE], y_train[0:BATCH_SIZE] used

python main.py --export
