import numpy as np
import random, math
import pandas
import matplotlib.pyplot as plt


def gaussianDataGen(mean, var):
    """
    Univariate gaussian data generator
    (Applying Box-Muller Method)
    """
    x = random.random()
    y = random.random()
    standard_norm = (-2 * math.log(x))**(1 / 2) * math.cos(
        2 * math.pi * y)  # standart Gaussian
    norm = standard_norm * math.sqrt(
        var) + mean  # standard_norm = (norm - mean) / std
    return norm


def genDataPoints(N, m_x, v_x, m_y, v_y):
    data_points_x = np.zeros((N, 3))
    # data_points_y = np.zeros((N, 1))
    for i in range(N):
        data_points_x[i][0] = gaussianDataGen(m_x, v_x)
        data_points_x[i][1] = gaussianDataGen(m_y, v_y)
        data_points_x[i][2] = 1
    return data_points_x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradientDescent(data_x, data_y):
    N = data_x.shape[0]
    w = np.zeros((3, 1), dtype='float64')
    design_matrix_x = data_x
    # print(design_matrix_x.shape)
    last_w = w.copy()
    #last_w = w
    learning_rate = 0.001
    while (1):
        '''
        gradient = x.T @ (yi - 1 / (1 + exp(-Xi@W)))
        '''
        gradient = design_matrix_x.T @ (data_y - sigmoid(design_matrix_x @ w))
        w = w + learning_rate * gradient
        if (np.linalg.norm(w - last_w) <= 0.01):
            break
        last_w = w.copy()
        # print(design_matrix_x.T @ gradient)
        # print(w)
        # print('\n')
    print('Gradient Descent: \n')
    print_result(w, design_matrix_x @ w, data_y)
    return design_matrix_x @ w


def newton_hessian(x, y, w):
    N = x.shape[0]
    diagonal_pivot = np.exp((-1) * x @ w) / ((1 + np.exp(-x @ w))**2)
    diagonal_pivot = diagonal_pivot.reshape(N)
    D = np.diag(diagonal_pivot)
    # print(D)
    # exit()
    return x.T @ D @ x


def newtonMethod(data_x, data_y):
    N = data_x.shape[0]
    design_matrix_x = data_x
    w = np.zeros((3, 1), dtype='float64')
    last_w = w.copy()
    while (1):
        # print("gino")
        '''
        Wn+1 = Wn - inv(Hessian_of_f) @ gradient_of_f
        '''
        gradient = design_matrix_x.T @ (data_y - sigmoid(design_matrix_x @ w))
        try:
            hessian_inverse = np.linalg.inv(
                newton_hessian(design_matrix_x, data_y, w))
        except np.linalg.LinAlgError:
            w = w + gradient
        else:
            w = w + hessian_inverse @ gradient
        if (np.linalg.norm(w - last_w) <= 0.01):
            break
        last_w = w.copy()
        # print(design_matrix_x.T @ gradient)
        # print(w)
        # print('\n')
    print("Newton's method: \n")
    print_result(w, design_matrix_x @ w, data_y)
    return design_matrix_x @ w


class Visualization:
    def __init__(self):
        self.graph_title = [
            'Ground truth', 'Gradient descent', "Newton's method"
        ]

    def draw(self, data_points, predict_label, subplot_idx):
        N = data_points.shape[0]
        plt.subplot(1, 3, subplot_idx + 1)
        plt.title(self.graph_title[subplot_idx])
        '''
        draw data points
        '''
        data_x = data_points[:, 0].reshape(N, 1)
        data_y = data_points[:, 1].reshape(N, 1)
        # cluster 1
        cluster_1_data_x = data_x[predict_label <= 0.5]
        cluster_1_data_y = data_y[predict_label <= 0.5]
        # cluster 2
        cluster_2_data_x = data_x[predict_label > 0.5]
        cluster_2_data_y = data_y[predict_label > 0.5]
        # plot
        plt.plot(cluster_1_data_x, cluster_1_data_y, '.', color='red')
        plt.plot(cluster_2_data_x, cluster_2_data_y, '.', color='blue')


def print_result(w, predict_label, ground_truth):
    N = predict_label.shape[0]
    print('w:')
    print(w)
    confusion_matrix = np.zeros((2, 2))
    # print(predict_label)
    # print(ground_truth)
    # exit()
    for i in range(N):
        if (predict_label[i] <= 0.5 and ground_truth[i] <= 0.5):
            confusion_matrix[0][0] += 1
        elif (predict_label[i] > 0.5 and ground_truth[i] <= 0.5):
            confusion_matrix[0][1] += 1
        elif (predict_label[i] <= 0.5 and ground_truth[i] > 0.5):
            confusion_matrix[1][0] += 1
        elif (predict_label[i] > 0.5 and ground_truth[i] > 0.5):
            confusion_matrix[1][1] += 1

    confusion_matrix_result = pandas.DataFrame(
        confusion_matrix,
        columns=['Predict cluster 1', 'Predict cluster 2'],
        index=['Is cluster 1', 'Is cluster 2'])
    print('\n')
    print(confusion_matrix_result)
    print(
        '\nSensitivity (Successfully predict cluster 1): ',
        confusion_matrix[0][0] /
        (confusion_matrix[0][0] + confusion_matrix[0][1]))
    print(
        'Specificity (Successfully predict cluster 2): ',
        confusion_matrix[1][1] /
        (confusion_matrix[1][0] + confusion_matrix[1][1]))


def logisticRegression(N, m_x1, v_x1, m_y1, v_y1, m_x2, v_x2, m_y2, v_y2):
    data_points_1 = genDataPoints(N, m_x1, v_x1, m_y1, v_y1)
    data_points_2 = genDataPoints(N, m_x2, v_x2, m_y2, v_y2)
    data_label_1 = np.zeros((N, 1))
    data_label_2 = np.ones((N, 1))
    data_points = np.concatenate((data_points_1, data_points_2))
    data_label = np.concatenate((data_label_1, data_label_2))
    '''
    ground truth
    '''
    graph = Visualization()
    graph.draw(data_points, data_label, 0)
    # print(data_points.shape)
    # print(data_label.shape)
    # exit()
    '''
    gradient descent
    '''
    gradient_predict = gradientDescent(data_points, data_label)
    graph.draw(data_points, gradient_predict, 1)
    '''
    Newton's method
    '''
    print('----------------------------------------')
    newton_predict = newtonMethod(data_points, data_label)
    graph.draw(data_points, newton_predict, 2)

    plt.show()


if __name__ == "__main__":
    logisticRegression(50, 1, 2, 1, 2, 3, 4, 3, 4)
    # logisticRegression(50, 1, 2, 1, 2, 10, 2, 10, 2)