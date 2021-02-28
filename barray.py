from array import array

class BArrayOs:

    def __init__(self):
        self._tensors = []

    def add_tensor(self, val):
        self._tensors.append(val)

    def write_to_file(self, path):
        with open(path, 'wb') as f:
            self._write_all(f)

    def _write_int_array(self, os, data):
        data = array('i', [int(x) for x in data])
        data.tofile(os)

    def _write_f32_array(self, os, data):
        data = array('f', [float(x) for x in data])
        data.tofile(os)

    def _write_tensor(self, os, val):
        # Write shape
        self._write_int_array(os, [len(val.shape)])
        self._write_int_array(os, val.shape)

        # Write data
        self._write_f32_array(os, val.ravel())

    def _write_all(self, os):
        # Write number of tensors
        self._write_int_array(os, [len(self._tensors)])

        # Write all tensors
        for t in self._tensors:
            self._write_tensor(os, t)



    @staticmethod
    def write_tensors_to_file(path, tensors):
        print('Exporting {} tensors to {}...'.format(len(tensors), path))
        bos = BArrayOs()
        for t in tensors:
            bos.add_tensor(t)
        bos.write_to_file(path)
