import torch
import numpy as np


def Norm_1_numpy(y):
    sum1 = 0
    for i in range(y.shape[0]):
        sum1 = sum1 + np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum1 / y.shape[0]


def Norm_1_torch(y):
    sum = 0
    for i in range(y.shape[0]):
        sum = sum + torch.max(torch.norm(y[i], p=1, dim=0))
    return sum / y.shape[0]


def Norm_1(y):
    sum = 0
    for i in range(y.shape[0]):
        sum = sum + np.linalg.norm(y[i], ord=1, keepdims=True)
    return sum / y.shape[0]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

