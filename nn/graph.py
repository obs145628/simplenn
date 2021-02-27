import numpy as np

from . import ops

class Graph:

    def __init__(self):
        self.lnodes = list()
        self.dnodes = dict()
        self._grads_cache = dict()


    ### Builder methods for all ops ###

    def build_const(self, val, name='const'):
        name = self._gen_unique_name(name)
        node = ops.ConstOp(name, val)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_placeholder(self, shape, name='placeholder'):
        name = self._gen_unique_name(name)
        node = ops.PlaceholderOp(name, shape)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_variable(self, init, name='variable'):
        name = self._gen_unique_name(name)
        node = ops.VariableOp(name, init)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_bias_add(self, lhs, rhs, name='bias_add'):
        name = self._gen_unique_name(name)
        node = ops.BiasAddOp(name, lhs, rhs)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_matmul(self, lhs, rhs, transpose_lhs=False, transpose_rhs=False, name='matmul'):
        name = self._gen_unique_name(name)
        node = ops.MatmulOp(name, lhs, rhs, transpose_lhs, transpose_rhs)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_multiply(self, lhs, rhs, name='multiply'):
        name = self._gen_unique_name(name)
        node = ops.MultiplyOp(name, lhs, rhs)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_relu(self, input, name='relu'):
        name = self._gen_unique_name(name)
        node = ops.ReluOp(name, input)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_relu_grad(self, input, dout, name='relu_grad'):
        name = self._gen_unique_name(name)
        node = ops.ReluGradOp(name, input, dout)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_reshape(self, input, out_shape, name='matmul'):
        name = self._gen_unique_name(name)
        node = ops.ReshapeOp(name, input, out_shape)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_softmax(self, input, name='softmax'):
        name = self._gen_unique_name(name)
        node = ops.SoftmaxOp(name, input)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_softmax_cross_entropy(self, labels, logits, name='softmax_cross_entropy'):
        name = self._gen_unique_name(name)
        node = ops.SoftmaxCrossEntropyOp(name, labels, logits)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_subtract(self, lhs, rhs, name='subtract'):
        name = self._gen_unique_name(name)
        node = ops.SubtractOp(name, lhs, rhs)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_sum(self, input, axis, name='sum'):
        name = self._gen_unique_name(name)
        node = ops.SumOp(name, input, axis)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node

    def build_tile(self, input, multiplier, name='tile'):
        name = self._gen_unique_name(name)
        node = ops.TileOp(name, input, multiplier)
        self.lnodes.append(node)
        self.dnodes[name] = node
        return node
    




    # Build a node that compute the gradient of output_node with respect to var_node
    # output_node must be a scalar
    # var_node can be any kind of node
    # All gradients nodes are cached and built only once
    def create_gradient_node(self, output_node, var_node):
        if len(output_node.shape) != 0:
            raise Exception('Can only compute the gradient of a scalar')
        if not var_node.is_pred_rec(output_node):
            # Could also return 0
            raise Exception('Gradient is 0: {} is not an ancestor of {}'.format(
                var_node.name, output_node.name))

        # Check if gradient is already computed
        res = self._find_in_grads_cache(output_node, var_node)
        if res is not None:
            return res

        # Check base case
        if output_node is var_node:
            #dX_dX = 1
            res = self.build_const(np.ones(()).astype(np.float32))
            self._insert_in_grads_cache(output_node, var_node, res)
            return res


        res = None #could also be set 0
        
        # There may be multiple paths from var_node to output_node, need to check them all
        for succ_node in var_node.succs:
            if not succ_node.is_pred_rec(output_node):
                continue

            # Compute gradient of output_node with respect to succ_node
            grad_succ = self.create_gradient_node(output_node, succ_node)
            # Use it to compute gradient of output_node with respect to var_node
            grad_var = succ_node._backward(self, grad_succ, succ_node.preds.index(var_node))
            assert grad_var.shape == var_node.shape

            # sum with paths found before
            if res is None:
                res = grad_var
            else:
                #res = self.build_add(res, grad_var)
                raise Exception('Gradient failed: multiple paths, AddOp not implemented')

        # Cache and returns computed gradient
        self._insert_in_grads_cache(output_node, var_node, res)
        return res

    
    # compute all node values in nodes
    # return list of values
    # feeds is the dictionary: <placeholder-name, placeholder-val>
    def run_ops(self, nodes, feeds):

        # Set placeholders
        for (ph_name, ph_val) in feeds.items():
            self.dnodes[ph_name]._placeholder_set(ph_val)

        # Eval all nodes
        res = [n._compute() for n in nodes]

        # Cleanup

        # Return results
        return res

    # Use a code-like representation to dump all the computations
    def dump_ops_code(self, nodes):
        print('\n=================================================')
        print('Compute nodes:', [n.name for n in nodes])
        computed = set()
        for n in nodes:
            self._dump_ops_code_rec(n, computed)
        print('=================================================\n')


    def _dump_ops_code_rec(self, node, computed):
        if node.name in computed:
            return

        for pred in node.preds:
            self._dump_ops_code_rec(pred, computed)

        print(node.get_code_str())
        computed.add(node.name)
        

    def _gen_unique_name(self, name):
        if name not in self.dnodes:
            return name
        
        idx = 2
        while True:
            ext_name = '{}_{}'.format(name, idx)
            if ext_name not in self.dnodes:
                return ext_name
            idx += 1


    def _get_grad_hash(self, output_node, var_node):
        return '[{}]|[{}]'.format(output_node.name, var_node.name)

    def _find_in_grads_cache(self, output_node, var_node):
        key = self._get_grad_hash(output_node, var_node)
        if key not in self._grads_cache:
            return None
        else:
            return self._grads_cache[key]

    def _insert_in_grads_cache(self, output_node, var_node, grad_node):
        key = self._get_grad_hash(output_node, var_node)
        assert key not in self._grads_cache
        self._grads_cache[key] = grad_node
