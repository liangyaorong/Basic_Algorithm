#coding:utf-8
import numpy as np
import pandas as pd

def dist(vet1, vet2):
    return np.sqrt(np.sum(np.power(vet1-vet2,2)))

# 给每个分类创建一个随机中心
def random_center(train_X, k):
    n,m = train_X.shape
    max = np.max(train_X, axis=0)
    min = np.min(train_X, axis=0)
    random_cent = np.zeros([k, m])
    for i in range(k):
        random_cent[i,:] = min + (max-min)*np.random.rand()
    return random_cent

def kMeans(train_X, k):
    m,n = train_X.shape
    label = np.zeros(m)

    center = random_center(train_X,k)

    while True:
        old_center = center
        for i in range(m):
            min_dist = np.inf
            for j in range(k):
                distance = dist(train_X[i,:], center[j,:])
                if distance<min_dist:
                    min_dist = distance
                    label[i] = j
        for classify in range(k):
            classify_train = train_X[label == classify]
            if len(classify_train) == 0:
                continue
            center[classify,:] = classify_train.mean(axis=0)

        if (pd.DataFrame(center).dropna(axis=0) == pd.DataFrame(old_center).dropna(axis=0)).all().all():
            break
    return label

if __name__=='__main__':
    vet1 = [1, 2]
    vet2 = [5.5, 6.5]
    vet3 = [6, 7]
    vet4 = [3, 2]
    vet_mat = np.array([vet1, vet2, vet3, vet4])

    print kMeans(vet_mat,3)
