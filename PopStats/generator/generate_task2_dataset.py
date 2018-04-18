import numpy as np
import matplotlib.pyplot as plt

plot = False


def generate_task2_dataset(L, N):

    np.random.seed(3)

    # initialization
    X = []               # inputs, bags representing the input distributions
    Y = np.zeros((L, ))  # output labels

    # X_params
    d = 2  # dimensions
    A = np.random.rand(d, d)  # randomly initialize Cholesky of Sigma
    X_params = np.linspace(0, np.pi, L)  # rotation angles

    # X, Y
    np.random.seed(None)

    for nL in range(L):

        A_nL = np.dot(rotation_matrix(X_params[nL]), A)

        # bag represents a rotated normal distribution
        X.append(np.dot(A_nL, np.random.randn(d, N)).T)

        # entropy of the first coordinate
        M = np.dot(A_nL, A_nL.T)
        s = M[0, 0]
        Y[nL] = 1.0 / 2 * np.log(2 * np.pi * np.exp(1) * s**2)

    if plot:
        plt.plot(X_params, Y)
        plt.xlabel('Rotation angle')
        plt.ylabel('Entropy of the first coordinate')
        plt.xlim([0, np.pi])
        plt.title('Entropy vs. Rotation angle')
        plt.show()

    return X, Y, X_params


def rotation_matrix(angle):
    C = np.cos(angle);
    S = np.sin(angle)
    R = np.array([[C, -S], [S, C]])
    return R


if __name__ == '__main__':

    generate_task2_dataset(2**7, 512)
