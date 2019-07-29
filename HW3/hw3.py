import numpy as np
import matplotlib.pyplot as plt


def gaussian_data_generator(mean, variance):

    def standard_normal():
        U, V = np.random.random_sample(2)
        return np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)

    return standard_normal() * np.sqrt(variance) + mean


def linear_data_generator(a, W):
    """
        a: variance
        W(list): weights
    """
    n = len(W)
    e = gaussian_data_generator(0, a)
    #e = 0
    x = np.random.uniform(-1, 1)
    X = np.array([x**i for i in range(n)])
    return x, W @ X + e


def sequential_estimator(mean, variance, true_params, data=[]):
    '''calculate new mean & variance by Welford's online learning Algorithm '''
    if len(data) == 0:
        new_data = gaussian_data_generator(true_params['mean'],
                                           true_params['variance'])
        print("Add data point: ", new_data)
        data.append(new_data)
        mean = new_data
        variance = 0

    while np.abs(mean - true_params['mean']) + np.abs(
            variance - true_params['variance']) > 0.001:
        new_data = gaussian_data_generator(true_params['mean'],
                                           true_params['variance'])
        print("Add data point: ", new_data)
        data.append(new_data)
        n = len(data)
        #variance = variance * (n - 2) / n - 1 + (new_data - mean)**2 / n
        variance = variance + (new_data - mean)**2 / n - variance / (n - 1)
        mean = mean + (new_data - mean) / n
        #print(np.mean(data))
        #print(np.var(data, ddof=1))

        print("Mean = {}    Variance = {}".format(mean, variance))
    print("Mean = {}    Variance = {}".format(mean, variance))
    '''
    if np.abs(mean - true_params['mean']) + np.abs(
            variance - true_params['variance']) > 0.2:
        mean, variance = sequential_estimator(mean, variance, true_params,
                                       data)
    '''
    return mean, variance


def baysianLR(precision, a, W):
    """Baysian Linear Regression
    
    """


def demo1():
    true_params = {'mean': 0, 'variance': 1}
    #data = [true_params['mean'] for i in range(3)]
    #mean = np.mean(data)
    #variance = np.var(data)
    mean = 0
    variance = 0
    mean, variance = sequential_estimator(mean, variance, true_params, data=[])
    #data = gaussian_data_generator(mean, variance, [])


def demo2():

    def update_plot(x, y, f):
        x_data = np.append(f.get_xdata(), x)
        y_data = np.append(f.get_ydata(), y)
        f.set_xdata(x_data)
        f.set_ydata(y_data)
        plt.draw()
        plt.pause(0.01)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    f, = ax.plot([], [], 'o')

    ax.set_ylim(-10, 20)
    ax.set_xlim(-2, 2)
    W = [1, 2, 3, 4]
    for i in range(1000):
        new_x, new_y = linear_data_generator(0, W)
        update_plot(new_x, new_y, f)

    plt.ioff()
    plt.show()


class Demo3Graph:

    def __init__(self):
        self.fig = plt.figure()

    def ground_truth(self, W, a):
        ground_ax = self.fig.add_subplot(2, 2, 1)
        X = []
        x_axis = []
        n = len(W)
        ground_ax.set_ylim(-10, 20)
        ground_ax.set_xlim(-2, 2)
        for x in np.arange(-2, 2, 0.1):
            x_axis.append(x)
            x_data = [x**i for i in range(n)]
            X.append(x_data)
        x_axis = np.array(x_axis)
        y_axis = np.array(X) @ np.array(W).T
        f, = ground_ax.plot(x_axis, y_axis, '-', color='black')
        ground_ax.plot(x_axis, y_axis + a, '-', color='red')
        ground_ax.plot(x_axis, y_axis - a, '-', color='red')

    def predict_result(self, X, mu, a, cov, y, n, subplot_param):
        ax = self.fig.add_subplot(subplot_param)
        predict_X = []
        predict_y = []
        x_axis = []
        ax.set_ylim(-10, 20)
        ax.set_xlim(-2, 2)
        # x axis
        for x in np.arange(-2, 2, 0.1):
            x_axis.append(x)
            predict_X.append([x**i for i in range(n)])
            predict_y.append(predict_X[-1] @ mu)

        # data_points
        ax.plot([X[i][1] for i in range(len(X))],
                y,
                'o',
                color='blue',
                markersize=3)

        ax.plot(x_axis, predict_y, '-', color='black')  # mean
        X = np.array(X)
        predict_y = np.array(predict_y)
        var = 1 / a + X[-1] @ cov @ X[-1].T
        ax.plot(
            x_axis, predict_y + var, '-',
            color='red')  # mean with positive variance
        ax.plot(
            x_axis, predict_y - var, '-',
            color='red')  # mean with negative variance


def demo3():
    """
        b: precision
    """

    W = [1, 2, 3]
    n = len(W)
    a = 1
    b = 1
    b_i = 1 / b
    prior_mean = np.array([0 for i in range(n)])
    prior_cov = b_i * np.identity(n)
    X = []
    y = np.array([])
    predictive_mean = 0
    predictive_variance = np.inf

    data_i = 0
    graph = Demo3Graph()
    graph.ground_truth(W, a)
    while True:
        data_i += 1
        new_x, new_y = linear_data_generator(a, W)
        x_data = [new_x**i for i in range(n)]  # 多項式
        X.append(x_data)
        X_np = np.array(X)
        y = np.append(y, new_y)
        S = np.linalg.inv(prior_cov)

        print("\nAdd data point ({}, {})".format(new_x, new_y))

        post_cov = np.linalg.inv(a * np.dot(X_np.T, X_np) + S)
        post_mean = post_cov @ (a * X_np.T @ y + S @ prior_mean)
        print("\nPosterior mean: \n", post_mean)
        print("\nPosterior variance \n", post_cov)

        predictive_mean = X_np[-1] @ prior_mean
        predictive_variance = 1 / a + X_np[-1] @ prior_cov @ X_np[-1].T
        print("\nPredictive distribution ~ N({}, {})".format(
            predictive_mean, predictive_variance))

        if data_i == 10:
            graph.predict_result(X, post_mean, a, post_cov, y, n, '223')
        elif data_i == 50:
            graph.predict_result(X, post_mean, a, post_cov, y, n, '224')

        if np.sum(np.abs(prior_cov - post_cov)) + np.sum(
                np.abs(prior_mean - post_mean)) < 0.001 and data_i >= 50:
            break
        else:
            prior_mean = post_mean
            prior_cov = post_cov

    graph.predict_result(X, post_mean, a, post_cov, y, n, '222')
    plt.show()


#demo1()
#demo2()
demo3()