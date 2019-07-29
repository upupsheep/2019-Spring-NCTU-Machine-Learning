import numpy as np


def test(b):
    b.append(1)
    b = 1


a = set([1, 2, 3])

for i in a:
    print(i)
    a = a.union(set([i - 1]))

print(a)