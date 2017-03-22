#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import csv


'''
最后一列为Y，前面是回归系数X
'''
def load_dataset(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split(' ')#自己设置分割标志
        float_line = map(float, curline)   #将每行映射成浮点数
        datamat.append(float_line)
    fr.close()
    return np.mat(datamat)



def linear_solve(dataset):
    m,n = np.shape(dataset)
    X = np.mat(np.ones((m,n)))
    X[:,1:n] = dataset[:,0:n-1]
    Y = dataset[:,-1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('this matrix is singular, cannot do inverse')
    beta = xTx.I * (X.T * Y)
    return beta,X,Y

#叶子点返回回归系数，误差以残差平方和来衡量---------------------------------------
def model_leaf(dataset):#叶子点返回的是回归系数
    beta,X,Y = linear_solve(dataset)
    return beta

def model_err(dataset):#误差测量为残差平方和
    beta,X,Y = linear_solve(dataset)
    y_hat = X * beta
    return sum(np.power(Y-y_hat,2))

#叶子点返回均值，误差以总平方和来衡量------------------------------------------------
def reg_leaf(dataset):
    return np.mean(dataset[:,-1])

def reg_err(dataset):
    return np.var(dataset[:,-1])*np.shape(dataset)[0]#总平方和， n*D(y)=sum(yi-mean(y))

#-----------------------------------------------------------------------------
def bin_split_dataset(dataset, feature, value):
    mat0 = dataset[np.nonzero(dataset[:,feature] <= value)[0], :]#取出feature<=value的行.即树中提示的val属于左集
    mat1 = dataset[np.nonzero(dataset[:,feature] > value)[0], :]
    return mat0, mat1#左分支和右分支

def choose_best_split(dataset, leaf_type = reg_leaf, err_type = reg_err, ops = (1,4)):
    tolS = ops[0]; tolN = ops[1]     #tolS是容许的误差下降值，tolN是切分的最少样本数（样本数（行数）若小于该值则不再分割）。要切分的越细致，tolS要调小，tolN也要调小。这是预剪枝
    if len(set(dataset[:,-1].T.tolist()[0])) ==1:  #如果所有y都一样，则返回该值（求均值也是该值），并说明没有最好的分割点
        return None, leaf_type(dataset)
    m,n = np.shape(dataset)
    S = err_type(dataset)#原始误差
    best_S = np.inf#默认情况数据。无论哪个分割点，其分支样本数都小于最小样本数，最后得到的仍是默认情况
    best_index = 0
    best_value = 0
    for feat_index in range(n-1):   #对列循环
        for split_val in set(dataset[:,feat_index].T.tolist()[0]):   #对列中所有取值循环（为了找出最好的列和列中最好的分割值）
            mat0,mat1 = bin_split_dataset(dataset,feat_index,split_val)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):#若有任意一支的样本数小于可容忍最小样本数，不进入误差计算，best_S,best_index,best_value不改变
                continue#跳出本次循环
            new_S = err_type(mat0) + err_type(mat1)
            if new_S < best_S:
                best_index = feat_index
                best_value = split_val
                best_S = new_S
    if (S - best_S) < tolS:#若分割后改善的误差小于可容忍的最小改善误差（默认情况也包含在里面），则采取不分割的策略，返回y的平均值，并说明无最好分割点
        return None,leaf_type(dataset)
    return best_index,best_value#若找到分割点满足最小样本条件，且其分割出来的最优误差改善情况满足最小可容忍改善误差，则返回该分割点


def create_tree(dataset, leaf_type = reg_leaf, err_type = reg_err, ops = (1,4)):#递归树的一般方法是先用 if设置好叶子点要返回的值，然后进入递归主程序
    feat,val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat == None:#若无满足条件的分割点，即不能再分割了，则返回该分割集的y均值。即说明该树的叶子点是均值。
        return val
    return_tree = {}
    return_tree['spInd'] = feat
    return_tree['spVal'] = val
    left_set, right_set = bin_split_dataset(dataset, feat, val)
    return_tree['left'] = create_tree(left_set, leaf_type, err_type, ops)
    return_tree['right'] = create_tree(right_set, leaf_type, err_type, ops)
    return return_tree


#------------------------------------后剪枝----------------------------------------------------------------------
def istree(obj):
    return (type(obj).__name__ == 'dict')

def get_mean(tree):#将该树坍塌成一个值。查找该树末端，若一结点两支均为叶子点，则将其两支的值合并（取均值），并返回上一层。上一层继续合并。最后整棵树完全合并成一个值。
    # if istree(tree['right']):
    #     tree['right'] = get_mean(tree['right'])
    # if istree(tree['left']):
    #     tree['left'] = get_mean(tree['left'])
    # return (tree['right'] + tree['left']) / 2.0
    if not istree(tree['left']) and not istree(tree['right']):
        return (tree['left']+tree['right'])/2.0

    if istree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if istree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):#若两边都不是树，直接判断是否合并，若两支至少一支是树，则先递归直到两边都是树，再判断是否合并。只适用与均值树
    if np.shape(test_data)[0] == 0:#若测试集为空，即该测试集在树下没有分类，则对该枝进行坍塌，并返回坍塌后的值（因为该值根本没啥用）
        return get_mean(tree)

    if (istree(tree['right']) or istree(tree['left'])):  # 若树至少有一支为树，则先对该测试数据按决策树进行分割，然后递归剪枝。有可能返回的还是树
        left_set, right_set = bin_split_dataset(test_data, tree['spInd'], tree['spVal'])  # 对测试集按树的方式切分
        if istree(tree['left']):
            tree['left'] = prune(tree['left'], left_set)  # 对左支递归更新，测试集为切分后的测试集左支
        if istree(tree['right']):
            tree['right'] = prune(tree['right'], right_set)  # 对右支递归

    if not istree(tree['left']) and not istree(tree['right']):#  对递归后  的两支，若都不是树，判断是否合并；若至少一支是树，返回树本身。对于叶子点，显然直接判断是否合并，else无意义
        left_set,right_set = bin_split_dataset(test_data,tree['spInd'],tree['spVal'])
        tree_mean = (tree['right'] + tree['left']) / 2.0
        error_merge = sum(np.power(test_data[:,-1] - tree_mean, 2))#假定以树的结果为均值，计算测试集合并后的方差
        error_no_merge = sum(np.power(left_set[:,-1] - tree['left'],2)) + sum(np.power(right_set[:,-1] - tree['right'],2))#假定以树的结果为均值，计算测试集不合并的方差
        if error_merge < error_no_merge:#若合并后误差变小，则合并该树的两个叶子。若合并后误差没有变小，则保持原来的两个叶子（均值）不变。递归会
            print 'merging'
            return tree_mean
        else:
            return tree
    else:
        return tree




#-------------------用树进行预测--------------------------------------------
def reg_tree_eval(model,in_data):
    return float(model)

def model_tree_eval(model,in_data):
    n = np.shape(in_data)[1] #注意，in_data必须是行矩阵
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = in_data
    return float(X*model)

def tree_forcast(tree,in_data,model_eval = reg_tree_eval):#输入的为一行的矩阵。若分到叶子点，则返回叶子点的预测数据，若未到叶子点，则判断属于左右支，然后调用自身进入左右支直到找到叶子点为止
    if not istree(tree):
        return model_eval(tree,in_data)

    if in_data[0,tree['spInd']] > tree['spVal']:
        return tree_forcast(tree['right'],in_data,model_eval)
    else:
        return tree_forcast(tree['left'],in_data,model_eval)


def create_forcast(tree,test_data,model_eval = reg_tree_eval):#对预测集每一样本进行预测并记录结果给yhat
    m = len(test_data)
    yhat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yhat[i,0] = tree_forcast(tree,np.mat(test_data[i]),model_eval)
    return yhat



reader = csv.reader(file('artist_1_tabulate.csv','r'))
data = []
for i in reader:
    i = map(float,i)
    data.append(i)
data = np.mat([[i[-1],i[0]] for i in data])


tree = create_tree(data,leaf_type = model_leaf, err_type = model_err,ops = (100000,4))
print tree
yhat =  create_forcast(tree,np.arange(20150801,20150830),model_eval=model_tree_eval)
print yhat
# tree_prune = prune(tree,np.mat([[15,-1]]))
# print tree_prune
# print get_mean(tree_prune)

# plt.figure(0)
# plt.plot(data[:,0].tolist(),data[:,1].tolist())
# plt.title(u'Regression Tree')
# plt.show()

