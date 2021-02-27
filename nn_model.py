import nn.model as model
from nn.optimizers import SGDOptimizer, SoftmaxCrossEntropyLoss

def build_nn_model(batch_size, learning_rate):
    model_base = model.Model(input_shape=(batch_size, 28, 28), layers=[
        model.FlattenLayer(),
        model.DenseLayer(128, activation='relu'),
        model.DenseLayer(32, activation='relu'),
        model.DenseLayer(10),
    ])

    model_probs = model.Model(base=model_base, layers=[
        model.SoftmaxLayer()
    ])

    optimizer = SGDOptimizer(model_base, SoftmaxCrossEntropyLoss(labels_shape=(batch_size, 10)),
                             lr=learning_rate)
    
    return model_base, model_probs, optimizer
