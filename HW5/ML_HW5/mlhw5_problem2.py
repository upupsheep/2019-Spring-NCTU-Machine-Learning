from libsvm.python.svmutil import *
from libsvm.python.svm import *
import numpy as np
import matplotlib.pyplot as plt


def gen_data(filename):
    return np.genfromtxt(filename, delimiter=',')


def train_with_linear_kernel(y, x):
    param = '-t 0 -h 0'
    model = svm_train(y, x, param)

    print('test:')
    p_label, p_acc, p_val = svm_predict(y, x, model)
    print(p_label)
    p_label = np.array(p_label)
    return p_label, model


def train_with_polynomial_kernel(y, x):
    param = '-t 1 -h 0'
    model = svm_train(y, x, param)

    print('test:')
    p_label, p_acc, p_val = svm_predict(y, x, model)
    print(p_label)
    p_label = np.array(p_label)
    return p_label, model


def train_with_RBF_kernel(y, x):
    param = '-t 2 -h 0'
    model = svm_train(y, x, param)

    print('test:')
    p_label, p_acc, p_val = svm_predict(y, x, model)
    print(p_label)
    p_label = np.array(p_label)
    return p_label, model


def train_with_precomputed_kernel(y, x):
    def linear_kernel(x1, x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K_train = np.zeros((n1, n2 + 1))
        K_train[:, 1:] = np.dot(x1, x2.T)
        K_train[:, 0] = np.arange(n1) + 1
        return K_train

    def RBF_kernel(x1, x2, gamma):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K_train = np.zeros((n1, n2 + 1))
        x1_norm = np.sum(x1**2, axis=-1)
        x2_norm = np.sum(x2**2, axis=-1)
        dist = x1_norm[:, None] + x2_norm[None, :] - 2 * np.dot(x1, x2.T)
        K_train[:, 1:] = np.exp(-gamma * dist)
        K_train[:, 0] = np.arange(n1) + 1
        return K_train

    def precomputed_kernel(x1, x2, gamma):
        linear_rbf = linear_kernel(x1, x2) + RBF_kernel(x1, x2, gamma)
        linear_rbf[:, 0] /= 2
        return linear_rbf

    param = '-t 4 -h 0'
    x_train = precomputed_kernel(x, x, 2**-5)
    model = svm_train(y, x_train, param)

    print('test: ')
    p_label, p_acc, p_val = svm_predict(y, x_train, model)
    print(p_acc[0])
    p_label = np.array(p_label)
    return p_label, model


class Visualization:
    def __init__(self, y, x):
        self.title = [
            'Linear kernel', 'Polynomial kernel', 'RBF kernel',
            'Linear + RBF kernel'
        ]
        self.y = y
        self.x = x

    def plot_svm_cluster(self, predict_label, graph_idx, sv_idx):
        n = int(predict_label.shape[0])
        shape_type = ['s', '^', 'x', 'd']
        marker_color = []
        marker_shape = []
        for i in range(n):
            # different color for different cluster
            if predict_label[i] == 0:
                marker_color.append('r')
            elif predict_label[i] == 1:
                marker_color.append('g')
            else:
                marker_color.append('b')

            # different shape for different support vector
            if i in sv_idx:
                marker_shape.append(shape_type[graph_idx])
            else:
                marker_shape.append('.')
        plt.subplot(2, 2, graph_idx + 1)
        plt.title(self.title[graph_idx])
        x1_axis = self.x.T[0]
        x2_axis = self.x.T[1]
        for i in range(n):
            plt.scatter(
                x1_axis[i],
                x2_axis[i],
                s=16,
                marker=marker_shape[i],
                c=marker_color[i])

    def show_graph(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    y_train = gen_data('Plot_Y.csv')
    x_train = gen_data('Plot_X.csv')

    linear_predict, linear_model = train_with_linear_kernel(y_train, x_train)
    poly_predict, poly_model = train_with_polynomial_kernel(y_train, x_train)
    rbf_predict, rbf_model = train_with_RBF_kernel(y_train, x_train)
    linear_rbf_predict, linear_rbf_model = train_with_precomputed_kernel(
        y_train, x_train)

    linear_sv_idx = np.array(linear_model.get_sv_indices()) - 1
    poly_sv_idx = np.array(poly_model.get_sv_indices()) - 1
    rbf_sv_idx = np.array(rbf_model.get_sv_indices()) - 1
    linear_rbf_sv_idx = np.array(linear_rbf_model.get_sv_indices()) - 1

    graph = Visualization(y_train, x_train)
    graph.plot_svm_cluster(linear_predict, 0, linear_sv_idx)
    graph.plot_svm_cluster(poly_predict, 1, poly_sv_idx)
    graph.plot_svm_cluster(rbf_predict, 2, rbf_sv_idx)
    graph.plot_svm_cluster(linear_rbf_predict, 3, linear_rbf_sv_idx)
    graph.show_graph()