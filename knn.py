import numpy as np

def dist(vet1, vet2):
    return np.sqrt(np.sum(np.power(vet1-vet2, 2)))

def knn(train_X, label, to_classify_vet):
    m,n = train_X.shape
    classify = -1
    distance = np.inf
    for i in range(m):
        dis = dist(train_X[i,:], to_classify_vet)
        if dis<distance:
            distance = dis
            classify = label[i]
    return classify

if __name__=='__main__':
    vet1 = [1, 2]
    vet2 = [5.5, 6.5]
    vet3 = [6, 7]
    vet4 = [3, 2]
    train_X = np.array([vet1, vet2, vet3, vet4])
    label = np.array([1, 2, 2, 1])
    to_classify_vet = np.array([7,6])
    print knn(train_X, label, to_classify_vet)