import random

class DatasetIterator:

    def __init__(self, ds):
        self.ds = ds
        self.data = list(range(0, self.ds.batches_count))
        random.shuffle(self.data)
        self.idx = 0

    def __next__(self):
        if self.idx == len(self.data):
            raise StopIteration

        beg = self.data[self.idx] * self.ds.batch_size
        end = beg + self.ds.batch_size
        self.idx += 1

        return (self.ds.X[beg:end], self.ds.y[beg:end])

class Dataset:

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.batches_count = len(self.X) // self.batch_size 

    def __iter__(self):
        return DatasetIterator(self)
