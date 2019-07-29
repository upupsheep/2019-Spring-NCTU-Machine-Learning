import numpy as np


def dataGen(filepath):
    # open file
    fp = open(filepath, 'r')

    # parsing input info: Ax = b
    A = []
    b = []
    for line in fp:
        test_data = line.strip().split(',')
        A.append(float(test_data[0]))  # x
        b.append(float(test_data[1]))  # y

    # end of parsing
    fp.close()
    A = np.transpose(np.matrix(A))  # transpose
    A = np.c_[A, np.ones(A.shape[0])]  # add dummy 1's
    b = np.transpose(np.matrix(b))  # transpose
    print(A.shape)
    print(b)


A = np.array([[1, 2], [1, 3]])
print(A * A)
print(np.matmul(A, A))
#print(A.reshape(A.shape[0], 1))
