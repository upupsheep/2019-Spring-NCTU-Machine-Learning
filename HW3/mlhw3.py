import numpy as np
import random, math
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


def linearDataGen(a, W):
    """
    a(variance)
    W(n*1 vector, weight)
    """
    n = len(W)  # basis of W
    e = gaussianDataGen(0, a)
    # y = WX + e
    xi = np.random.uniform(-1, 1)
    X = np.array([xi**i for i in range(n)])
    y = W.T @ X + e
    return xi, y


def sequentialEstimator(m, s):
    print('Data point source function: N({}, {})'.format(m, s))
    print('')
    """
    Sequential estimate the mean and variance
    Apply Welford's online algorithm
    Data is given from the univariate gaussian data generator (1.a).
    """
    n = 1
    new_data = gaussianDataGen(m, s)
    sample_mean = new_data / n
    sample_var = 0.0

    old_sample_mean = sample_mean
    old_sample_var = sample_var

    while True:
        # New data in! Update sample_mean and sample_variance
        n += 1
        new_data = gaussianDataGen(m, s)
        sample_mean = old_sample_mean + (new_data - old_sample_mean) / n
        sample_var = old_sample_var + (
            new_data - old_sample_mean)**2 / n - old_sample_var / (n - 1)
        print('Add data point: ', new_data)
        print('Mean = {}    Variance = {}\n'.format(sample_mean, sample_var))
        # check converge
        if abs(sample_mean - m) < 0.01 and abs(sample_var - s) < 0.01:
            break
        # update old mean and variance
        old_sample_mean = sample_mean
        old_sample_var = sample_var


class Visualize:
    def __init__(self):
        self.graph_title = [
            'Ground truth', 'Predict result', 'After 10 incomes',
            'After 50 incomes'
        ]

    def groundTruth(self, a, W):
        n = len(W)
        x_point = np.array([])
        y_point = np.array([])
        for x in np.arange(-2, 2, 0.1):
            x_vector = np.array([np.power(x, i) for i in range(n)])
            y = W.T @ x_vector
            x_point = np.append(x_point, x)
            y_point = np.append(y_point, y)

        plt.subplot(2, 2, 1)
        plt.xlim((-2, 2))
        plt.ylim((-15, 25))
        plt.title(self.graph_title[0])
        plt.plot(x_point, y_point, '-', color='black')
        plt.plot(x_point, y_point + a, '-', color='red')
        plt.plot(x_point, y_point - a, '-', color='red')

    def predictResult(self, x_vector, y_vector, mean, cov, a, n, subplot_idx):
        # generate data points

        x_vector = np.array(x_vector)
        x_mean = np.array([])
        predict_x = []
        predict_y = []
        var = []
        for x in np.arange(-2, 2, 0.1):
            x_mean = np.append(x_mean, x)
            predict_x.append([x**i for i in range(n)])
            predict_y.append(predict_x[-1] @ mean)
            # plot variance
            var.append(
                1 / a +
                np.array(predict_x[-1]) @ cov @ np.array(predict_x[-1]).T)

        plt.subplot(2, 2, subplot_idx)
        plt.xlim((-2, 2))
        plt.ylim((-15, 25))
        plt.title(self.graph_title[subplot_idx - 1])
        # plot data points
        plt.plot(
            [x_vector[i][1] for i in range(len(x_vector))],
            y_vector,
            'o',
            markersize=3)

        # plot mean
        plt.plot(x_mean, predict_y, '-', color='black')

        # plot variance
        predict_y = np.array(predict_y)
        var = np.array(var).reshape((40, 1))
        plt.plot(x_mean, predict_y + var, '-', color='red')
        plt.plot(x_mean, predict_y - var, '-', color='red')
        # print(x_vector[-1] @ cov @ x_vector[-1].T)
        print(x_mean.shape)
        print(predict_y.shape)
        print(var.shape)
        print((np.add(predict_y, var)).shape)
        # exit()


def baysianLinearRegression(b, a, W):
    """
    b: precision prior(covariance inverse)
    S = inv(prior_cov), m = prior_mean
    """
    n = len(W)
    graph = Visualize()
    graph.groundTruth(a, W)

    # repeat update prior until the posterior probability converges.
    S = b * np.eye(n)
    m = np.zeros((n, 1))
    data_x = []
    data_y = []

    in_cnt = 0
    while True:

        # New data in!!!
        new_data_x, new_data_y = linearDataGen(a, W)
        in_cnt += 1
        # new_data_x, new_data_y = fake_datas[i]
        # i += 1
        new_x_vector = np.array([new_data_x**i for i in range(n)])
        data_x.append(new_x_vector)
        data_y.append([new_data_y])
        new_x_vector = np.array(data_x)
        new_y_vector = np.array(data_y)

        # print(new_x_vector)
        # exit()
        print('Add data point ({}, {})\n'.format(new_data_x, new_data_y))
        '''
        *** calculate new posterior ***
        post_cov_inv = a X.T X + S
        post_mean = post_cov (a X.T y + Sm)
        '''
        # calculate new posterior
        post_S = a * new_x_vector.T @ new_x_vector + S
        post_mean = np.linalg.inv(post_S) @ (
            a * new_x_vector.T @ new_y_vector + S @ m)
        # calculate predictive mean and variance
        predict_mean = new_x_vector[-1] @ m
        predict_var = 1 / a + new_x_vector[-1] @ S.T @ new_x_vector[-1].T
        print('Posterior mean:')
        print(post_mean)
        print('Posterior variance:')
        print(np.linalg.inv(post_S))
        print('predict mean:')
        # print(predict_mean)
        print('\nPredictive distribution ~ N({}, {})'.format(
            predict_mean[0], predict_var))
        # print('data_x: ')
        # print(data_x[-1])
        print('-----------------------------------------------')

        if in_cnt == 10:
            graph.predictResult(data_x, data_y, post_mean,
                                np.linalg.inv(post_S), a, n, 3)

        elif in_cnt == 50:
            graph.predictResult(data_x, data_y, post_mean,
                                np.linalg.inv(post_S), a, n, 4)

        # check convergency
        if np.linalg.norm(np.linalg.inv(post_S) - np.linalg.inv(S)
                          ) < 0.001 and np.linalg.norm(post_mean - m) < 0.001:
            break

        # update next prior as posterior
        S = post_S
        m = post_mean

    graph.predictResult(data_x, data_y, post_mean, np.linalg.inv(post_S), a, n,
                        2)
    # plt.show()


if __name__ == "__main__":
    mean = 3.0
    var = 5.0
    # sequentialEstimator(mean, var)

    b = 1
    a = 1
    W = np.array([1, 2, 3, 4])
    baysianLinearRegression(b, a, W)
    plt.show()