import numpy as np
from itertools import chain
from scipy.misc import imread
import os
import glob
import random
import matplotlib.pyplot as plt


def loadImgData():
    n = 400
    filenames = [img for img in glob.glob("att_faces/s*/*.pgm")]
    m = [[] for i in range(n)]
    for i in range(n):
        m[i] = list(chain.from_iterable(imread(filenames[i])))

    m = np.matrix(m)  # (400, 10304)
    # print(m.shape)
    return m


class pca:
    def __init__(self, k):
        self.k = k

    def mean(self, data):
        return np.mean(data, axis=1)

    def scatter_matrix(self, data):
        """ S = sum((Xk-m)@(Xk-m)^T) / n, where k=1,...,n """
        return np.cov(data, bias=True)

    def find_k_largest_eigenvalues(self, cov):
        print('Calculating eigen values and vectors...')
        k = self.k
        eigen_value, eigen_vector = np.linalg.eig(cov)
        sorting_index = np.argsort(-eigen_value)
        eigen_value = eigen_value[sorting_index]
        eigen_vector = eigen_vector.T[sorting_index]
        return eigen_value[0:k], (eigen_vector[0:k])

    def transform(self, W, data):
        # return W @ data
        return W.T @ W @ data

    def pca_main(self, data):
        # data => (d,n) (10304, 400)
        ### mean ###

        mean = self.mean(data)  # (10304,)
        print(mean.shape)
        data = data.copy() - mean  # n*d
        ### S(covariance) ###
        S = self.scatter_matrix(data.T)  #(10304, 10304) -> (400, 400)
        print(S.shape)
        ### eigenvector & eigenvalue -> principle components ###
        eigen_value, eigen_vector = self.find_k_largest_eigenvalues(
            S)  # (25, 400)
        # print(eigen_vector.shape)
        eigen_vector = (data @ eigen_vector.T).T
        print('eigen_value:')
        print(eigen_value)
        print('eigen_vector:')
        print(eigen_vector.shape)
        ### Now W is eigen_vector (25, 10304) ###
        '''
        fig, axes = plt.subplots(5, 5)
        # idx = np.random.choice(25, 10, replace=False)
        # print(idx)
        for i in range(25):
            axes[int(i / 5), i % 5].imshow(
                eigen_vector[i].reshape(112, 92), cmap="gray")
            # axes[1, i].imshow(
            #     transformed_data[random_idx].reshape(112, 92), cmap="gray")
        plt.show()
        '''
        transformed_data = self.transform(eigen_vector, data)
        # np.savetxt('transformed.txt', np.imag(transformed_data))
        transformed_data = np.real(transformed_data)
        print(transformed_data)
        return transformed_data.T


if __name__ == "__main__":
    k = 25

    face_matrix = loadImgData()  # (400, 10304)

    pca_model = pca(k)
    transformed_data = pca_model.pca_main(
        face_matrix.T)  # (400, 10304) -> (400, 10304)

    print('transformed data: ', transformed_data.shape)

    fig, axes = plt.subplots(2, 10)
    idx = np.random.choice(400, 10, replace=False)
    print(idx)
    for i, random_idx in enumerate((idx)):
        axes[0, i].imshow(
            face_matrix[random_idx].reshape(112, 92), cmap="gray")
        axes[1, i].imshow(
            transformed_data[random_idx].reshape(112, 92), cmap="gray")
    plt.show()
