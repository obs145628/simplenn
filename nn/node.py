import numpy as np

def shape_len(shape):
    res = 1
    for n in shape: res *= n
    return res

class Node:

    def __init__(self, name, out_shape, inputs):
        self.name = name
        self.shape = list(out_shape)
        self.preds = list(inputs)
        self.succs = list()
        self.size = shape_len(self.shape)
        self.rank = len(self.shape)

        for p in self.preds:
            p.succs.append(self)
        
        self._node_val = None

    # Operation name (eg add, matmul)
    def opname(self):
        raise Exception('Node.opname() not implemented for {}'.format(self.name))

    # Returns dict<string, string> list of attributes. Used only to print instruction
    def get_attrs_dict(self):
        return dict()

    # Returns a string representing the current operation
    def get_code_str(self):
        res = '%{} = {}('.format(self.name, self.opname())
        for (idx, pred) in enumerate(self.preds):
            if idx != 0:
                res += ', '
            res += '%{}'.format(pred.name)
        res += ')'

        attrs = self.get_attrs_dict()
        if len(attrs) > 0:
            res += ' {' + ', '.join(['{}: {}'.format(key, val)
                                     for key, val in attrs.items()])  + '}' 

        res += ': ('
        for (idx, pred) in enumerate(self.preds):
            if idx != 0:
                res += ', '
            res += 'x'.join([str(n) for n in pred.shape])
        res += ')'

        res += ' -> ({})'.format('x'.join([str(n) for n in self.shape]))

        return res
        
        

    # Returns true if self is an ancestor of x
    def is_pred_rec(self, x):
        if self is x:
            return True
        for succ in self.succs:
            if succ.is_pred_rec(x):
                return True
        return False
        

    # Function called when one of the preds value is invalidated
    # this happens when a variable is updated, or a placeholder has a new value
    # should set self._node_val to None if it invalidates this node value
    def _input_changed(self, invalidated_pred):
        self._node_val = None


    # Call this to invalidate the node
    # Only called by variable / placeholder node when the node value changed
    def _invalidate(self):
        self._invalidate_succs()

    def _invalidate_succs(self):
        # Go through all successors and invalidate already valid node
        for s in self.succs:
            if s._node_val is not None:
                s._input_changed(self)
                if s._node_val is None:
                    s._invalidate_succs()


    # Called when node is used as a placeholder to set a value
    # Implemented only by placeholder Node
    def _placeholder_set(self, val):
        raise Exception('Not a placeholder node')

    # Called to evaluate the value of the current Node
    # Value must be stored in self._node_val
    # All predecessors value are already evaluated
    def _evaluate(self):
        raise Exception('_evluate() not impplemented for {}'.format(self.name))
        
    # Return the node value that must already be evaluated
    def _val(self):
        if self._node_val is None:
            raise Exception('Node not evaluated')
        return self._node_val


    # Compute all the predecessors before evaluting current node
    def _compute(self):
        if self._node_val is not None:
            return self._node_val

        for p in self.preds:
            p._compute()

        self._evaluate()
        res = self._val()
        assert res.dtype == np.float32
        if list(res.shape) != list(self.shape):
            raise Exception('Compute for node {} ({}) returns tensor {}, should be {}'.format(
                self.name, self.opname(), res.shape, self.shape))
        

        return res


    # Add operations to the graph to compute dL_dpreds[i] given Dl_dself = dout
    # Return the built node
    def _backward(self, graph, dout, i):
        raise Exception('Gradient not implemented for {} ({}) w.r.t. input #{}:{} ({}))'.format(
            self.name, self.opname(), i, self.preds[i].name, self.preds[i].opname()))
        
