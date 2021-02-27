from .dataset import Dataset

class Loss:

    def __init__(self, labels_shape):
        self._labels_shape = labels_shape
        self._labels_node = None
        self._preds_node = None
        self._loss_node = None
        self._imodel = None

    # self._labels_node and self._preds_node are already defined
    # this function must be overload to compute loss from these 2 nodes
    # after the call, self._loss_node must be set to the graph node to compute the loss
    def _build(self):
        raise Exception('Loss._init() not implemented')

# Performs SoftmaxCrossentropy with logits and dense labels
class SoftmaxCrossEntropyLoss(Loss):

    def __init__(self, labels_shape):
        super().__init__(labels_shape)

    def prepare(self, model):
        self._imodel = model._imodel
        graph = self._imodel.graph

        self._labels_node = graph.build_placeholder(self._labels_shape, name='labels')
        self._preds_node = model._output_node
        if self._preds_node is None:
            raise Exception('Missing output node for model')

        self._build()
        if self._loss_node is None:
            raise Exception('Missing output node for loss')

    def _build(self):
        graph = self._imodel.graph
        sce = graph.build_softmax_cross_entropy(self._labels_node, self._preds_node)
        self._loss_node = graph.build_sum(sce, axis=0, name='softmax_cross_entropy_sum')

# Create an optimizer with the model that returns the predictions,
# and a specific loss function
class SGDOptimizer:

    def __init__(self, model, loss_fn, lr=1e-3):
        self._model = model
        self._loss = loss_fn
        self._lr = lr
        self._imodel = model._imodel
        

        # Init Loss
        self._loss.prepare(model)

        self._grads_nodes = None

    def _build_grads_nodes(self):
        graph = self._imodel.graph
        self._grads_nodes = []
        
        for w in self._imodel.trainable_vars:
            grad = graph.create_gradient_node(self._loss._loss_node, w)
            self._grads_nodes.append(grad)
        return self._grads_nodes

    # compute the loss value from the labels and the model input
    def loss(self, labels, input):
        nodes = [self._loss._loss_node]
        feeds = {self._loss._labels_node.name: labels,
                 self._model._input_node.name: input}
        result = self._imodel.graph.run_ops(nodes, feeds)
        return result[0]

    # compute the loss using batches
    def compute_full_loss(self, X_test, y_test, batch_size):
        ds = Dataset(X_test, y_test, batch_size)
        loss_node = self._loss._loss_node
        total_loss = 0
            
        for (batch_idx, (X_batch, y_batch)) in enumerate(ds):
            feeds = {self._loss._labels_node.name: y_batch,
                     self._model._input_node.name: X_batch}
            batch_loss = self._imodel.graph.run_ops([loss_node], feeds)[0]              
            total_loss += batch_loss

        return total_loss / len(X_test)
        

    def train(self, X_train, y_train, X_test, y_test, num_epochs, batch_size, dump_code=False):

        ds = Dataset(X_train, y_train, batch_size)

        train_vars = self._imodel.trainable_vars
        all_nodes = self.grads_nodes() + [self._loss._loss_node]
        if dump_code:
            self._imodel.graph.dump_ops_code(all_nodes)

        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for (batch_idx, (X_batch, y_batch)) in enumerate(ds):
                feeds = {self._loss._labels_node.name: y_batch,
                         self._model._input_node.name: X_batch}
                results = self._imodel.graph.run_ops(all_nodes, feeds)
                grads = results[0:-1]
                batch_loss = results[-1]
                
                epoch_loss += batch_loss

                # Update weights
                for (grad_val, var) in zip(grads, train_vars):
                    var.set_val(var.get_val() - self._lr *  grad_val)


            epoch_loss = epoch_loss / len(X_train)
            test_loss = self.compute_full_loss(X_test, y_test, batch_size)
            print('Epoch {}/{}: train loss = {}, test loss = {}'.format(
                epoch+1, num_epochs, epoch_loss, test_loss))
        

    # Return all graph nodes to compute the gradient of each weight
    def grads_nodes(self):
        if self._grads_nodes is None:
            self._build_grads_nodes()
        return self._grads_nodes

    # Compute the value of the gradient for every weight of the model
    def compute_gradients(self, labels, input):
        nodes = self.grads_nodes()
        feeds = {self._loss._labels_node.name: labels,
                 self._model._input_node.name: input}
        result = self._imodel.graph.run_ops(nodes, feeds)
        return result

    # dump code to compute all the gradients
    def dump_gradients_code(self):
        nodes = self.grads_nodes()
        self._imodel.graph.dump_ops_code(nodes)
