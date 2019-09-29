import pyximport
pyximport.install()
import fastloop
import numpy as np
import timeit
import pprint

D = 5
N = 1000
X = np.array([np.random.rand(D) for d in range(N)])
beta = np.random.rand(N)
theta = 10

def func():
    f = fastloop.rbf_network(X, beta, theta)
    print(f)

n = 1
t = timeit.timeit(stmt=func, number=n)
pprint.pprint(t)
