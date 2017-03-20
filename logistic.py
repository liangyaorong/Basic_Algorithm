#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import random

def load_data():
    data_mat = [[1.0,1.0,1.0],[1.0,1.0,1.05],[1.0,1.5,1.3],[1.0,0.1,0.05],[1.0,0.08,0.12],[1.0,0.11,0.2]]
    labels = [1,1,1,0,0,0]
    return data_mat,labels

def sigmoid(x):
    y = 1.0/(1 + np.exp(-x))
    return y

#梯度上升找参数
def grad_ascent(data_mat,labels_mat):
    data_mat = np.mat(data_mat)
    labels_mat = np.mat(labels_mat).transpose()
    m,n = np.shape(data_mat)
    alpha = 0.1
    max_cycles = 500
    weights = np.ones((n,1))
    for i in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = (labels_mat - h)
        weights = weights + alpha * data_mat.transpose() * error
    return weights


#随机梯度上升找参数
def stoc_grad_ascent(data_mat,labels_mat):
    data_mat = np.array(data_mat)
    labels_mat = np.array(labels_mat)
    m,n = np.shape(data_mat)
    alpha = 0.1
    max_cycles = 500
    weights = np.ones(n)
    for i in range(max_cycles):
        for j in range(m):
            rand_index = int(random.uniform(0,m))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = labels_mat[rand_index] - h
            weights = weights + alpha * error * data_mat[rand_index]
    return np.mat(weights).transpose()


def classify_logistic(in_X,weights):
    X = [1]
    X.extend(in_X)
    weights = np.mat(weights)
    X = np.mat(X)
    y = X * weights
    if y > 0: judge = 1
    else: judge = 0
    return judge


data,labels = load_data()
weights = grad_ascent(data,labels)
judge = classify_logistic([0.6,0.6],weights)
print judge

