import sys

EPS = 1e-6

class Tester:

    def __init__(self):
        self.nb_total = 0
        self.nb_valid = 0

    def check_tensors(self, name, ref_arr, my_arr):
        self.nb_total += 1

        if ref_arr.dtype != my_arr.dtype:
            self._err(name, 'Expected array of type {}, got {}'.format(ref_arr.dtype,
                                                                       my_arr.dtype))
            return
        
        if ref_arr.shape != my_arr.shape:
            self._err(name, 'Expected array of shape {}, got {}'.format(ref_arr.shape,
                                                                        my_arr.shape))
            return

        diff = (ref_arr.ravel() - my_arr.ravel()) @ (ref_arr.ravel() - my_arr.ravel())

        if diff > EPS:
            self._err(name, 'Array differs ({}:{}): diff = {}'.format(
                my_arr.shape, my_arr.dtype, diff))
            return

        self._succ(name, ref_arr, my_arr, diff)
        self.nb_valid += 1

    def end(self):
        print('\n\n')
        print('========== Tests Summarry ==========')
        print('Passed {}/{} tests ({}%)'.format(self.nb_valid,
                                                self.nb_total,
                                                (self.nb_valid/self.nb_total)*100))
        print('{} tests failed'.format(self.nb_total - self.nb_valid))

        is_ok = self.nb_valid == self.nb_total

        if is_ok:
            print('Success !')
        else:
            print('Faillure !')

        print('====================================')

        sys.exit(0 if is_ok else 1)
        

    def _err(self, name, msg):
        print('[KO]: {}: {}'.format(name, msg))

    def _succ(self, name, ref_arr, my_arr, diff):
        print('[OK]: {} ({}:{}): diff = {}'.format(name, my_arr.shape, my_arr.dtype, diff))

    

    
