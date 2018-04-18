import os
import numpy as np
from generate_task1_dataset import generate_task1_dataset

generators = [generate_task1_dataset]


def save_dataset(L, N, fname, generator):
    X, Y, X_params = generator(L, N)
    np.savez_compressed(fname, X=X, Y=Y, X_params=X_params, L=L, N=N)


def generate_dataset():

    for i, generator in enumerate(generators):

        odir = os.path.join('data', 'task{}'.format(i+1))

        # number of sets
        L = 512

        # number of points per set
        N = 512

        # generate test dataset
        save_dataset(L, N, os.path.join(odir, 'test'), generator)

        # generate validation dataset
        save_dataset(L, N, os.path.join(odir, 'val'), generator)

        # generate truth for plotting
        save_dataset(2**14, 1, os.path.join(odir, 'truth'), generator)

        # generate training dataset of different size:
        for logL in range(7, 18):
            L = 2 ** logL
            print(L)
            save_dataset(L, N, os.path.join(odir, 'data_{}'.format(logL)), generator)

        print('Data generated for task {}'.format(i+1))

    print('Done')


if __name__ == '__main__':

    generate_dataset()