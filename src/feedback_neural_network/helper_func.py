import random
import math


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1/0 + math.exp(-x))


def sigmod_derivate(x):
    return x * (1 - x)
