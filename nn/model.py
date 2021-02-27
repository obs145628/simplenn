import numpy as np

from .graph import Graph

class InternalModel:

    def __init__(self):
        self.graph = Graph()
        self.trainable_vars = []

    def add_var(self, name, init):
        v = self.graph.build_variable(init, name='weight_{}'.format(name))
        self.trainable_vars.append(v)
        return v


class Model:

    def __init__(self, input_shape=None, base=None, layers=[]):
        self._layers = []
        self._imodel = None
        self._input_node = None
        self._output_node = None
        next_node = None

        if base is None:
            self._imodel = InternalModel()
            
            if input_shape is None:
                raise Exception('input_shape required to build a new model')
            self._input_node = self._graph().build_placeholder(shape=input_shape,
                                                              name='input')
            next_node = self._input_node
        else:
            self._imodel = base._imodel
            self._input_node = base._input_node
            self._layers = list(base._layers)
            next_node = base._output_node



        for layer in layers:
            layer._imodel = self._imodel
            layer._input_node = next_node
            self._layers.append(layer)
            layer._build()
            if layer._output_node is None:
                raise Exception("Output node not defined by layer")
            next_node = layer._output_node


        self._output_node = next_node

    def _graph(self):
        return self._imodel.graph


    def forward(self, val, dump_code=False):
        nodes = [self._output_node]        
        feeds = {self._input_node.name: val}

        if dump_code:
            self._graph().dump_ops_code(nodes)
        
        result = self._graph().run_ops(nodes, feeds)
        return result[0]

    def __call__(self, val, dump_code=False):
        return self.forward(val, dump_code)

    def assign_weights(self, weights):
        for w, var in zip(weights, self._imodel.trainable_vars):
            var.set_val(w)

class Layer:

    def __init__(self):
        self._imodel = None
        self._input_node = None
        self._output_node = None

    # Called when layer ready to be built
    # self._input_node is set by the caller
    # this method must create graph nodes and set self._output_node
    def _build(self, imodel):
        raise Exception('Layer._build not implemented')


def activation_by_name(prefix, graph, input_node, act_name):
    node_name = '{}_{}'.format(prefix, act_name)
    if act_name == 'relu':
        return graph.build_relu(input_node, name=node_name)
    else:
        raise Exception('Unknown activation function `{}`'.format(act_name))

# y = activation(Wx+b)
class DenseLayer(Layer):

    def __init__(self, output_units, activation=None, init_weights=(None,None)):
        super().__init__()
        self._output_units = output_units
        self._activation = activation
        self._initW = init_weights[0]
        self._initb = init_weights[1]

    def _build(self):
        in_shape = self._input_node.shape
        if len(in_shape) != 2:
            raise Exception('DenseLayer: expected 2D input')
        in_units = in_shape[1]
        out_units = self._output_units
        graph = self._imodel.graph

        if self._initW is None:
            self._initW = np.random.randn(in_units, out_units).astype(np.float32)
        if self._initb is None:
            self._initb = np.zeros([out_units], dtype=np.float32)
        
        w = self._imodel.add_var('dense_W', self._initW)
        b = self._imodel.add_var('dense_b', self._initb)

        logits = graph.build_matmul(self._input_node, w, name='dense_matmul')
        logits = graph.build_bias_add(logits, b, name='dense_bias_add')
        if self._activation is not None:
            logits = activation_by_name('dense', graph, logits, self._activation)

        assert logits.shape[0] == in_shape[0] and logits.shape[1] == out_units
        self._output_node = logits
        
class FlattenLayer(Layer):

    def __init__(self):
        super().__init__()

    def _build(self):
        in_shape = self._input_node.shape
        if len(in_shape) < 2:
            raise Exception('Flatten: expected input with more than 1D')

        item_size = 1
        for x in in_shape[1:]:
            item_size *= x

        self._output_node = self._imodel.graph.build_reshape(self._input_node,
                                                             (in_shape[0], item_size),
                                                             name='flatten_reshape')
                                                             


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def _build(self):
        self._output_node = self._imodel.graph.build_softmax(self._input_node)


class ArgmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def _build(self):
        self._output_node = self._imodel.graph.build_argmax(self._input_node)
