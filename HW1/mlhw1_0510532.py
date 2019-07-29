import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def LUdecomposition(a):
    n = a.shape[0]
    l = np.zeros(a.shape)
    u = np.zeros(a.shape)
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            # summation of l[i][j]*u[j][k]
            sum = 0.0
            for j in range(i):
                sum += l[i][j] * u[j][k]
            # Evaluating U[i][k]
            u[i][k] = a[i][k] - sum

        # Lower Triangular
        for k in range(i, n):
            if (i == k):
                # main pivot 1
                l[i][i] = 1
            else:
                # summation of l[k][j]*u[j][i]
                sum = 0.0
                for j in range(i):
                    sum += l[k][j] * u[j][i]
                # Evaluating L[k][i]
                l[k][i] = (a[k][i] - sum) / u[i][i]

    # return
    return l, u


def LUinverse(l, u):
    # LUx = I
    # Ly = I
    # y = Ux = invert(L)
    n = l.shape[0]
    I = np.eye(n)  # Identity Matrix
    y = np.zeros(l.shape)
    x = np.zeros(l.shape)
    # solve Ly = I:(forward-substitution)
    for k in range(n):
        for i in range(n):
            # summation of l[i][j]*y[j][k]
            sum = 0.0
            for j in range(i):
                sum += l[i][j] * y[j][k]
            # Evaluating y[i][k]
            y[i][k] = I[k][i] - sum

    # solve Ux = y:(backward-substitution)
    for k in range(n):
        for i in range(n - 1, -1, -1):
            # summation of u[i][j]*x[j][k]
            sum = 0.0
            for j in range(i + 1, n):
                sum += u[i][j] * x[j][k]
            # Evaluating x[i][k]
            x[i][k] = (y[i][k] - sum) / u[i][i]

    # return inverse of LU
    return x


def LSE(A, lambd, b):
    # return: invers(A^TA+lambda I)(A^T)b
    A_gram = np.matmul(np.transpose(A), A)  # (A^T)A
    I = np.eye(A_gram.shape[0])  # Identity Matrix
    A_gram_lamdb = A_gram + lambd * I
    l, u = LUdecomposition(A_gram_lamdb)
    A_inverse = LUinverse(l, u)  # x = inverse of A_gram_lamdb
    return np.matmul(np.matmul(A_inverse, np.transpose(A)), b)


def gradient(A, b, x):
    # Gradient: 2*A^T*A*x - 2*A^T*b
    return 2 * np.matmul(np.matmul(np.transpose(A), A), x) - 2 * np.matmul(
        np.transpose(A), b)


def hessian(A, b, x):
    # Hessian: 2*A^T*A
    return 2 * np.matmul(np.transpose(A), A)


def NewtonMethod(A, b, x):
    # gradient and hessian
    gradient_x = gradient(A, b, x)
    hessian_x = hessian(A, b, x)
    # inverse of hessian
    l, u = LUdecomposition(hessian_x)
    hessian_x_inv = LUinverse(l, u)
    # h = f'(x) / f''(x) = Hf(x)^-1 * Gf(x)
    h = np.matmul(hessian_x_inv, gradient_x)
    while np.linalg.norm(h) >= 0.0001:
        # gradient and hessian
        gradient_x = gradient(A, b, x)
        hessian_x = hessian(A, b, x)
        # inverse of hessian
        l, u = LUdecomposition(hessian_x)
        hessian_x_inv = LUinverse(l, u)
        # h = f'(x) / f''(x) = Hf(x)^-1 * Gf(x)
        h = np.matmul(hessian_x_inv, gradient_x)

        # x(i+1) = x(i) - f'(x) / f''(x)
        x = x - h
    return x


def dataGen(filepath, n_base):
    # parsing input data: x,y => Ax = b
    A = []
    b = []
    x = []
    y = []
    with open(filepath, 'r') as fp:
        for line in fp:
            test_data = line.strip().split(',')
            # x
            x.append(float(test_data[0]))
            y.append(float(test_data[1]))
            # b: [y0,y1,y2...]^T
            b.append([float(test_data[1])])
            # A: # [x0,x1,x2...]^T
            A.append([
                np.power(float(test_data[0]), i)
                for i in reversed(range(n_base))
            ])

    # end of parsing
    A = np.array(A)
    b = np.array(b)
    x = np.array(x)
    y = np.array(y)
    # print(A.shape)
    # print(x.shape)
    A_train, A_test, b_train, b_test, x_train, x_test, y_train, y_test = train_test_split(
        A, b, x, y, test_size=0.3, random_state=10)
    # print(A_train.shape)
    # print(A_test.shape)

    # return
    return A_train, A_test, b_train, b_test, x_train, x_test, y_train, y_test


if __name__ == '__main__':
    n = 3
    lambd = 1000

    ### generate A, b: ###
    A_train, A_test, b_train, b_test, x_input, x_test_input, y_input, y_test_input = dataGen(
        'testfile.txt', n)

    ### LSE: ###
    LSE_result = LSE(A_train, lambd, b_train)
    LSE_error = np.power(
        np.linalg.norm(np.matmul(A_test, LSE_result) - b_test), 2)

    output_str = 'Fitting line: '
    output_str += str(LSE_result[0][0]) + 'X^' + str(n - 1)
    for i in range(1, n):
        if LSE_result[i][0] >= 0:
            output_str += ' +'
        output_str += ' ' + str(LSE_result[i][0]) + 'X^' + str(n - i - 1)
    print("LSE:")
    print(output_str)
    print("Total error: ", LSE_error)
    print("\n")

    ### Newton's Method: ###
    x0 = np.zeros((n, 1))
    Newton_result = NewtonMethod(A_train, b_train, x0)
    Newton_error = np.power(
        np.linalg.norm(np.matmul(A_test, Newton_result) - b_test), 2)

    output_str = 'Fitting line: '
    output_str += str(Newton_result[0][0]) + 'X^' + str(n - 1)
    for i in range(1, n):
        if Newton_result[i][0] >= 0:
            output_str += ' +'
        output_str += ' ' + str(Newton_result[i][0]) + 'X^' + str(n - i - 1)
    print("Newton's Method:")
    print(output_str)
    print("Total error: ", Newton_error)

    ### Visualization: ###
    diff_x = max(x_input) - min(x_input)
    # LSE:
    x_LSE = np.arange(min(x_input) - 1.0, max(x_input) + 1.0, diff_x / 1000)
    y_LSE = 0
    for i in range(n):
        y_LSE += LSE_result[i] * np.power(x_LSE, n - i - 1)
    # Newton:
    x_Newton = x_LSE
    y_Newton = 0
    for i in range(n):
        y_Newton += Newton_result[i] * np.power(x_Newton, n - i - 1)
    # plot
    plt.subplot(2, 1, 1)
    # plt.plot(x_input, y_input, 'ro')
    plt.plot(x_input, y_input, 'ro')
    plt.plot(x_LSE, y_LSE)
    plt.ylabel('LSE')

    plt.subplot(2, 1, 2)
    # plt.plot(x_input, y_input, 'ro')
    plt.plot(x_input, y_input, 'ro')
    plt.plot(x_Newton, y_Newton)
    plt.ylabel('Newton')
    plt.show()
