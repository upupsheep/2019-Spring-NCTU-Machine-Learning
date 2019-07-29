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


class LDA:
    def __init__(self, k, label):
        self.k = k
        self.label_min = int(np.min(label))
        self.label_max = int(np.max(label))
        self.class_num = int(np.max(label)) - int(np.min(np.min(label))) + 1

    def class_mean(self, data, label):
        class_mean = []
        for i in range(self.label_min, self.label_max + 1):
            class_mean.append(np.mean(data[label == i], axis=0))
        return np.array(class_mean)

    def overall_mean(self, data):
        return np.mean(data, axis=0)

    def within_class_scatter(self, data, label):
        """ sum of each class scatter """
        d = data.shape[1]  # 784
        within_class_scatter = np.zeros((d, d))
        for i in range(self.label_min, self.label_max + 1):
            within_class_scatter += np.cov(data[label == i].T)
        return np.array(within_class_scatter)

    def in_between_class_scatter(self, data, label, class_mean, overall_mean):
        """ sum(nj * (mj-m)@(mj-m)^T), where j=class index """
        class_data_cnt = []
        for i in range(self.label_min, self.label_max + 1):
            class_data_cnt.append(list(label).count(i))
        class_data_cnt = np.array(class_data_cnt)

        d = data.shape[1]  # 784
        in_between_class_scatter = np.zeros((d, d))
        for i in range(self.class_num):
            print(i, ':')
            # print(class_mean[i])
            # print(overall_mean)
            class_mean_col = class_mean[i].reshape(d, 1)
            overall_mean_col = overall_mean.reshape(d, 1)
            tmp = (class_mean_col - overall_mean_col) @ (
                class_mean_col - overall_mean_col).T
            print(tmp.shape)
            print('-------------')
            in_between_class_scatter += class_data_cnt[i] * (
                class_mean_col - overall_mean_col) @ (
                    class_mean_col - overall_mean_col).T
        in_between_class_scatter = np.array(in_between_class_scatter)
        return in_between_class_scatter

    def find_k_largest_eigenvalues(self, cov):
        k = self.k
        eigen_value, eigen_vector = np.linalg.eig(cov)
        sorting_index = np.argsort(-eigen_value)
        eigen_value = eigen_value[sorting_index]
        eigen_vector = eigen_vector.T[sorting_index]
        return eigen_value[0:k], (eigen_vector[0:k])

    def transform(self, W, data):
        return W @ data

    def lda_main(self, data, label):
        ### overall mean ###
        overall_mean = self.overall_mean(data)  # (784,)
        print(overall_mean.shape)
        # exit()
        ### calculate class mean ###
        class_mean = self.class_mean(data, label)  # (5,784)
        print(class_mean.shape)
        ### within-class scatter matrix ###
        within_class_s = self.within_class_scatter(data, label)  # (784, 784)
        print('within_class:')
        print(within_class_s.shape)
        ### in-between-class scatter matrix ###
        in_between_class_s = self.in_between_class_scatter(
            data, label, class_mean, overall_mean)
        print('in_between_class:')
        print(in_between_class_s.shape)
        # print(in_between_class_s)
        # np.savetxt('in_between_s.txt', in_between_class_s)
        #### eigenvalues & eigenvectors -> first k largest ###
        eigen_value, eigen_vector = self.find_k_largest_eigenvalues(
            np.linalg.pinv(within_class_s) @ in_between_class_s)
        print('eigen_vector:')
        print(eigen_vector.shape)
        print(eigen_vector)
        ### Now W is eigen_vector (2, 784) ###
        transformed_data = self.transform(np.real(eigen_vector), data.T)
        print('transformed_data:')
        print(transformed_data)
        return transformed_data


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
    label = loadLabel('mnist_label.csv')  #(5000,)

    lda_model = LDA(k, label)
    transformed_data = lda_model.lda_main(data_point, label)

    graph = Visualization()
    graph.plot(transformed_data, label)
