import numpy as np
import matplotlib.pyplot as plt
import os


def loadData(filepath):
    if os.path.exists('mnist_X.npy'):
        data = np.load('mnist_X.npy')
    else:
        data = np.loadtxt(filepath, delimiter=',')
        np.save('mnist_X', data)
    # print(data.shape)
    return data


def loadLabel(filepath):
    if os.path.exists('mnist_lable.npy'):
        data = np.load('mnist_lable.npy')
    else:
        data = np.loadtxt(filepath)
        np.save('mnist_label', data)
    # print(data.shape)
    return data


class pca:
    def __init__(self, k):
        self.k = k

    def mean(self, data):
        return np.mean(data, axis=1)

    def scatter_matrix(self, data):
        """ S = sum((Xk-m)@(Xk-m)^T) / n, where k=1,...,n """
        return np.cov(data, bias=True)

    def find_k_largest_eigenvalues(self, cov):
        k = self.k
        eigen_value, eigen_vector = np.linalg.eig(cov)
        sorting_index = np.argsort(-eigen_value)
        eigen_value = eigen_value[sorting_index]
        eigen_vector = eigen_vector.T[sorting_index]
        return eigen_value[0:k], (eigen_vector[0:k])

    def transform(self, W, data):
        return W @ data

    def pca_main(self, data):
        ### mean ###
        mean = self.mean(data)  # (784,)
        print(mean.shape)
        ### S(covariance) ###
        S = self.scatter_matrix(data)  #(784, 784)
        print(S.shape)
        ### eigenvector & eigenvalue -> principle components ###
        eigen_value, eigen_vector = self.find_k_largest_eigenvalues(S)
        print('eigen_value:')
        print(eigen_value)
        print('eigen_vector:')
        print(eigen_vector.shape)
        ### Now W is eigen_vector (2, 784) ###
        transformed_data = self.transform(eigen_vector, data)
        # np.savetxt('transformed.txt', np.imag(transformed_data))
        print(np.real(transformed_data))
        return np.real(transformed_data)


class Visualization:
    def __init__(self):
        pass

    def plot(self, data, label):
        n = int(label.shape[0])
        color_list = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
        marker_color = []
        for i in range(n):
            marker_color.append(color_list[int(label[i])])
        x1_axis = data[0]
        x2_axis = data[1]
        for i in range(n):
            plt.scatter(x1_axis[i], x2_axis[i], s=16, c=marker_color[i])
        plt.show()


if __name__ == "__main__":
    k = 2
    data_point = loadData('mnist_X.csv')  # (5000 * 784)

    pca_model = pca(k)
    transformed_data = pca_model.pca_main(data_point.T)
    print(transformed_data.shape)

    label = loadLabel('mnist_label.csv')  # (5000,)
    graph = Visualization()
    graph.plot(transformed_data, label)
