import numpy as np

from .node import Node

def shape_len(shape):
    res = 1
    for n in shape: res *= n
    return res

def shape_str(shape):
    return 'x'.join([str(n) for n in shape])


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

def log_softmax(x):
    x_max = np.max(x, axis=1).reshape(-1, 1)
    logsum = np.log(np.sum(np.exp(x - x_max), axis=1)).reshape(-1, 1)
    return x - x_max - logsum

def softmax_cross_entropy_with_logits(labels, logits):
    return -np.sum(labels * log_softmax(logits), axis=1)





# Constant tensor, never changes
class ConstOp(Node):

    def __init__(self, name, val):
        if val.dtype != np.float32:
            raise Exception('Only f32 type supported')
        
        super().__init__(name, val.shape, [])

        self._node_val = val

    def get_attrs_dict(self):
        res = {'shape': shape_str(self.shape)}
        if self._node_val.size < 10:
            res['val'] = '{}'.format(self._node_val).replace('\n', '')
        return res

    def opname(self):
        return 'const'

    def val(self):
        return self._node_val


# Placeholder node
# Value is set when running the graph
class PlaceholderOp(Node):

    def __init__(self, name, shape):
        super().__init__(name, shape, [])

    def opname(self):
        return 'placeholder'

    def get_attrs_dict(self):
        return {
            'shape': shape_str(self.shape)
        }

    def _placeholder_set(self, val):
        if val.dtype != np.float32:
            raise Exception('Only f32 type supported')
        
        if not np.array_equal(self._node_val, val):
            self._node_val = val
            self._invalidate()

    def _evaluate(self):
        raise Exception('Missing value for placeholder `{}`'.format(self.name))


# Variable node
# Value can be read, written any time
class VariableOp(Node):

    def __init__(self, name, init):
        super().__init__(name, init.shape, [])
        self._node_val = init

    def opname(self):
        return 'variable'

    def get_attrs_dict(self):
        return {
            'shape': shape_str(self.shape)
        }

    def get_val(self):
        return self._node_val

    def set_val(self, new_val):
        if list(new_val.shape) != list(self.shape):
            raise Exception('Cannot assign value {} to variable {}'.format(
                new_val.shape, self.shape))

        if new_val.dtype != np.float32:
            raise Exception('Only f32 type supported')
        
        if not np.array_equal(self._node_val, new_val):
            self._node_val = new_val
            self._invalidate()


class BiasAddOp(Node):

    def __init__(self, name, lhs, rhs):
        lshape = lhs.shape
        rshape = rhs.shape
        if len(lshape) != 2 or len(rshape) != 1:
            raise Exception('lhs and rhs must be respectively 2D and 1tensors')
        if rshape[0] != lshape[1]:
            raise Exception('Invalid size for rhs: lhs={}, rhs={}'.format(lshape, rshape))
        
        super().__init__(name, lshape, [lhs, rhs])

    def opname(self):
        return 'bias_add'

    def lhs(self):
        return self.preds[0]

    def rhs(self):
        return self.preds[1]

    def _evaluate(self):
        self._node_val = np.add(self.lhs()._val(), self.rhs()._val().reshape(1, -1))

    def _backward(self, graph, dout, i):
        if i == 0:
            return dout
        else:
            return graph.build_sum(dout, axis=0, name='bias_add_grad_sum')

class MatmulOp(Node):

    def __init__(self, name, lhs, rhs, transpose_lhs, transpose_rhs):
        if lhs.rank != 2 or rhs.rank != 2:
            raise Exception('lhs and rhs must be 2D tensors: lhs={}, rhs={}'.
                            format(lhs.shape, rhs.shape))

        lshape = lhs.shape
        rshape = rhs.shape
        if transpose_lhs:
            lshape = [lshape[1], lshape[0]]
        if transpose_rhs:
            rshape = [rshape[1], rshape[0]]
        

        if lshape[1] != rshape[0]:
            raise Exception('Invalid matrices sizes: lhs={}, rhs={}'.format(
                lhs.shape, rhs.shape))
        
        super().__init__(name, [lshape[0], rshape[1]], [lhs, rhs])

        self._transpose_lhs = transpose_lhs
        self._transpose_rhs = transpose_rhs

    def opname(self):
        return 'matmul'

    def get_attrs_dict(self):
        return {
            'transpose_lhs': str(self.transpose_lhs()),
            'transpose_rhs': str(self.transpose_rhs()),
        }

    def lhs(self):
        return self.preds[0]

    def rhs(self):
        return self.preds[1]

    def transpose_lhs(self):
        return self._transpose_lhs

    def transpose_rhs(self):
        return self._transpose_rhs

    def _evaluate(self):
        lhs = self.lhs()._val()
        rhs = self.rhs()._val()

        lhs = lhs.T if self.transpose_lhs() else lhs
        rhs = rhs.T if self.transpose_rhs() else rhs
        
        self._node_val = np.matmul(lhs, rhs)

    def _backward(self, graph, dout, i):
        if i == 0:
            return graph.build_matmul(dout, self.rhs(), transpose_rhs=True,
                                      name='matmul_grad_lhs_matmul')
        else:
            return graph.build_matmul(self.lhs(), dout, transpose_lhs=True,
                                      name='matmul_grad_rhs_matmul')

class MultiplyOp(Node):

    def __init__(self, name, lhs, rhs):
        if lhs.shape != rhs.shape:
            raise Exception('lhs {} and rhs {} must be of same shape'.format(
                lhs.shape, rhs.shape))        
        super().__init__(name, lhs.shape, [lhs, rhs])

    def opname(self):
        return 'multiply'

    def lhs(self):
        return self.preds[0]

    def rhs(self):
        return self.preds[1]

    def _evaluate(self):
        self._node_val = np.multiply(self.lhs()._val(), self.rhs()._val())

class ReluOp(Node):

    def __init__(self, name, input):
        super().__init__(name, input.shape, [input])

    def opname(self):
        return 'relu'

    def input(self):
        return self.preds[0]

    def _evaluate(self):
        x = self.input()._val()
        self._node_val = np.maximum(self.input()._val(), 0.)

    def _backward(self, graph, dout, i):
        return graph.build_relu_grad(self.input(), dout, name='relu_grad')

class ReluGradOp(Node):

    def __init__(self, name, input, dout):
        if list(input.shape) != list(dout.shape):
            raise Exception('ReluGrad operands {} and {} must have the same shape'.format(
                input.shape, dout.shape))
        
        super().__init__(name, input.shape, [input, dout])

    def opname(self):
        return 'relu_grad'

    def input(self):
        return self.preds[0]

    def dout(self):
        return self.preds[1]

    def _evaluate(self):
        x_prime = np.where(self.input()._val() > 0, 1.0, 0.0).astype(np.float32)
        self._node_val = self.dout()._val() * x_prime

class ReshapeOp(Node):

    def __init__(self, name, input, out_shape):

        if input.size != shape_len(out_shape):
            raise Exception('Incompatible shapes: {} ({}) can\'t be reshaped to {}({})'
                            .format(input.shape, input.size,
                                    out_shape, shape_len(out_shape)))

        
        super().__init__(name, out_shape, [input])
        self._out_shape = out_shape

    def opname(self):
        return 'reshape'

    def get_attrs_dict(self):
        return {
            'out_shape': shape_str(self.out_shape())
        }

    def input(self):
        return self.preds[0]

    def out_shape(self):
        return self._out_shape

    def _evaluate(self):
        self._node_val = np.reshape(self.input()._val(), self.out_shape())

class SoftmaxOp(Node):

    def __init__(self, name, input):
        if len(input.shape) != 2:
            raise Exception('Softmax expects 2D tensor, got {}'.format(input.shape))
        super().__init__(name, input.shape, [input])

    def opname(self):
        return 'softmax'

    def input(self):
        return self.preds[0]

    def _evaluate(self):
        self._node_val = softmax(self.input()._val())

class SoftmaxCrossEntropyOp(Node):

    def __init__(self, name, labels, logits):
        if len(labels.shape) != 2 or len(logits.shape) != 2:
            raise Exception('Label and logits for SoftmaxCrossEntropy must be 2D tensors')
        if labels.shape[0] != logits.shape[0] or labels.shape[1] != logits.shape[1]:
            raise Exception('Label and logits shape differ: {} <> {}'.format(
                labels.shape, logits.shape))

        super().__init__(name, (labels.shape[0],), [labels, logits])

    def opname(self):
        return 'softmax_cross_entropy'

    def labels(self):
        return self.preds[0]

    def logits(self):
        return self.preds[1]

    def _evaluate(self):
        labels = self.labels()._val()
        logits = self.logits()._val()
        self._node_val = softmax_cross_entropy_with_logits(labels, logits)

    def _backward(self, graph, dout, i):
        if i == 0:
            #grad not implemented for labels
            return super()._backward(graph, dout, i)

        sx = graph.build_softmax(self.logits(), name='softmax_cross_entropy_grad_softmax')
        sxc = graph.build_subtract(sx, self.labels(), name='softmax_cross_entropy_grad_sub')

        dout = graph.build_reshape(dout, (dout.size, 1),
                                   name='softmax_cross_entropy_grad_reshape')
        dout = graph.build_tile(dout, multiplier=(1, self.labels().shape[1]),
                                name='softmax_cross_entropy_grad_tile')

        return graph.build_multiply(sxc, dout, name='softmax_cross_entropy_mul')

class SubtractOp(Node):

    def __init__(self, name, lhs, rhs):
        if lhs.shape != rhs.shape:
            raise Exception('lhs {} and rhs {} must be of same shape'.format(
                lhs.shape, rhs.shape))        
        super().__init__(name, lhs.shape, [lhs, rhs])

    def opname(self):
        return 'subtract'

    def lhs(self):
        return self.preds[0]

    def rhs(self):
        return self.preds[1]

    def _evaluate(self):
        self._node_val = np.subtract(self.lhs()._val(), self.rhs()._val())

# Sum that reduces only on one axis
class SumOp(Node):

    def __init__(self, name, input, axis):
        if input.rank < 1:
            raise Exception('Cannot sum a scalar')
        if axis < 0 or axis >= input.rank:
            raise Exception('Axis {} must be in [0, {}]'.format(axis, input.rank-1))

        shape = list(input.shape)[:axis] + list(input.shape)[axis+1:]
        super().__init__(name, shape, [input])
        self._axis = axis

    def opname(self):
        return 'sum'

    def get_attrs_dict(self):
        return {
            'axis': str(self.axis())
        }

    def input(self):
        return self.preds[0]

    def axis(self):
        return self._axis

    def _evaluate(self):
        self._node_val = np.sum(self.input()._val(), axis=self.axis())

    def _backward(self, graph, dout, i):
        axis = self.axis()
        shape = list(self.input().shape)
        shape[axis] = 1
        multiplier = [1] * self.input().rank
        multiplier[axis] = self.input().shape[axis]
    
        dout = graph.build_reshape(dout, shape,
                                   name='backward_sum_0_reshape')
        return graph.build_tile(dout, multiplier,
                                name='backward_sum_0_tile')


class TileOp(Node):

    def __init__(self, name, input, multiplier):
        multiplier = list(multiplier)
        if len(input.shape) != len(multiplier):
            raise Exception('Multiplier {} has different length than input shape {}'.format(
                multiplier, input.shape))

        shape = [d * m for (d, m) in zip(input.shape, multiplier)]
        
        super().__init__(name, shape, [input])
        self._multiplier = multiplier

    def opname(self):
        return 'tile'

    def get_attrs_dict(self):
        return {
            'multiplier': str(self.multiplier())
        }

    def input(self):
        return self.preds[0]

    def multiplier(self):
        return self._multiplier

    def _evaluate(self):
        self._node_val = np.tile(self.input()._val(), self.multiplier())
