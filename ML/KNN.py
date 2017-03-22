# -*- coding:utf-8 -*-

'''
作者：梁耀荣
日期：2016年7月31日
注意：数据文件格式csv,且最后一列为分类标签
'''





import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import operator

'''
knn.py 为k近邻分类器
inX为需要分类的变量，为数组形式
dataset为训练集
labels 为训练集对应的标签，为数组形式
'''

# 从文件导入数据---------------------------------------------------------
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    lines = [line.strip().split(',')  for line in arrayOfLines]
    lines = [[float(value) for value in line] for line in lines]
    numberOfLine = len(lines)
    lines = np.array(lines)
    val = len(lines[0])-1
    returnMat = lines[:,0:val]
    classLabelsVector = lines[:,-1]
    fr.close()
    return returnMat,classLabelsVector


# 归一化-----------------------------------------------------------------
def normalize(dataset):
    min_data = dataset.min(0)
    max_data = dataset.max(0)
    ranges = max_data-min_data
    m = dataset.shape[0]
    norm_dataset = (dataset-np.tile(min_data,(m,1)))/np.tile(ranges,(m,1))
    return norm_dataset


# 分类-------------------------------------------------------------------
def classify(inX,dataSet,labels,k):
    #计算欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet                                              #tile将inX重复(dataSetSize*1)矩阵次
    distance = np.sqrt((diffMat**2).sum(axis = 1))                                              #axis=0：每列数据相加 axis = 1：每行数据相加
    #给距离排序
    sortedDistIndicies = distance.argsort()                                                     #argsort()返回的是数组值按从小到大排列的下标
    classCount = {}                                                                             #创建一个空字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]                                              #找出下标对应的标签
        if voteIlabel not in classCount.keys():
            classCount[voteIlabel] = 0
        classCount[voteIlabel] += 1
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),reverse = True)                      #先把字典中键值对提出来，再以字典的第二个域（标签的数量）排序，最后反序（由大到小）
    return sortedClassCount[0][0]                                                               #先将sorted()后的结果拿出第一对，再拿出这一对里的第一个


# 测试，计算错误率-------------------------------------------------------
def test(dataset,labels):
    testing_percent = 0.10
    k = 3
    dataset = normalize(dataset)
    m = dataset.shape[0]
    testing_num = int(math.ceil(m*testing_percent))
    errorCount = 0.0  
    for i in range(testing_num):
        predict_lable = classify(dataset[i,:],dataset[testing_num:,:],labels[testing_num:],k)
#        print predict_lable,labels[i]                                                          #把每个预测和实际的标签打印出来
        if (predict_lable!=labels[i]):
            errorCount += 1
    return errorCount/float(testing_num)

#----------------------便利函数----------------------------------------
def classify_KNN(filename,inX,k):
    data,labels=file2matrix(filename)
    data = normalize(data)
    pred = classify(inX, data, labels, k)
    error = test(data,labels)
    return pred,error
    
# 入口-----------------------------------------------------------------
pred,error = classify_KNN('points.csv',[1,0.9],2)
print '分类结果： %s'%pred
print '分类器错误率： %s'%error
