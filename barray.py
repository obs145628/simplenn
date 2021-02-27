from array import array

class BArrayOs:

    def __init__(self):
        self._tensors = []

    def add_tensor(self, val):
        self._tensors.append(val)

    def write_to_file(self, path):
        with open(path, 'wb') as f:
            self._write_all(f)
        
    def _write_shape(self, os, shape):
        data = array('i', [int(x) for x in shape])
        data.tofile(os)

    def _write_tensor(self, os, val):
        self._write_shape(os, val.shape)
        data = array('f', [float(x) for x in val.ravel()])
        data.tofile(os)

    def _write_all(self, os):
        self._write_shape(os, [len(self._tensors)])
        for t in self._tensors:
            self._write_tensor(os, t)



    @staticmethod
    def write_tensors_to_file(path, tensors):
        print('Exporting {} tensors to {}...'.format(len(tensors), path))
        bos = BArrayOs()
        for t in tensors:
            bos.add_tensor(t)
        bos.write_to_file(path)
