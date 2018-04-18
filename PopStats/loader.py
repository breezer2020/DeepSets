import pdb
import numpy as np
from tqdm import tqdm, trange
from sklearn.utils import shuffle


class DataIterator(object):
    def __init__(self, fname, batch_size, shuffle=False):

        self.fname = fname
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load data
        with np.load(fname) as f:
            self.L = np.asscalar(f['L'].astype('int32'))
            self.N = np.asscalar(f['N'].astype('int32'))
            self.t = f['X_params']
            self.X = f['X'].astype('float32')  # (L, N, d)
            self.d = self.X.shape[-1]
            self.y = f['Y'].astype('float32')

        assert len(self.y) >= self.batch_size, \
            'Batch size larger than number of training examples'
            
    def __len__(self):
        return len(self.y)//self.batch_size

    def get_iterator(self, loss=0.0):
        if self.shuffle:
            self.X, self.y, self.t = shuffle(self.X, self.y, self.t)
        return tqdm(self.next_batch(),
                    desc='Train loss: {:.4f}'.format(loss),
                    total=len(self), mininterval=1.0, ncols=80)
                    
    def next_batch(self):
        start = 0
        end = self.batch_size
        while end <= self.L:
            yield self.X[start:end], self.y[start:end]
            start = end
            end += self.batch_size
