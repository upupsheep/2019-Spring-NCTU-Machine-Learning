import numpy as np
import math


def Psuccess(data):
    success = data.count('1')
    total = len(data)
    return success, total, success / total


def Combin(N, m):
    return math.factorial(N) / (math.factorial(N - m) * math.factorial(m))


def Beta(a, b):
    return math.factorial(a + b - 1) / (
        math.factorial(a - 1) * math.factorial(b - 1))


def Likelihood(p, N, m):
    return Combin(N, m) * np.power(p, m) * np.power(1 - p, N - m)


if __name__ == '__main__':
    filename = 'testfile.txt'
    a = 10
    b = 1

    with open(filename) as f:
        case = 1
        for line in f:
            # line.rstrip('\n')
            line = line.rstrip()
            success, total, p_success = Psuccess(line)
            likelihood = Likelihood(p_success, total, success)
            print("case ", case, ":", line)
            print("Likelihood: ", likelihood)
            print("Beta prior: a = ", a, "b = ", b)
            a = success + a
            b = total - success + b
            print("Beta posterior: a = ", a, "b = ", b)
            print('\n')
            case += 1
