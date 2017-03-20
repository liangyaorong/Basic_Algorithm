# -*- coding:utf-8 -*-

#####################################################################################################
# 名称：决策树
# 作者：梁耀荣
# 日期：2016年7月31日
# 操作：使用classify_Tree()函数，输入训练数据文件和待分类向量，得到分类判断
# 注意：训练数据文件第一行为变量名称，最后一列为分类判断，且格式为csv。待分类向量每个元素均为字符串
# 适用：标称型分类
#####################################################################################################

import math
import operator
import numpy as np


# --------------------创建数据----------------------------------------------------
'''
dataset,labels均是列表（不是矩阵），labels为特征变量对应的标签
dataset中最后一列是分类标志
'''

def create_dataset(filename):
    fr = open(filename)
    data = [lines.strip().split(',') for lines in fr.readlines()]
    dataset = data[1:]
    labels = data[0]
    fr.close()
    return dataset,labels


# ----------------计算数据集香农熵-------------------------------------------
'''
找出每个标签的数量，其占总数量的百分比即为其频率
数据集中所有标签的香农熵之和为该数据集的香农熵
香农熵的计算公式：2^x = 1/p -> x = -log(p,2)
'''
def calc_Shannon_entropy(dataset):
    num_entries = len(dataset)
    label_count = {}
    for featvet in dataset:
        current_label = featvet[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1        
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key])/num_entries
        shannon_ent -= prob * math.log(prob,2)
    return shannon_ent


# -------------------划分数据集--------------------------------------------------
'''
axis为第几个特征
value为该特征取值
得到的是某一特征按某一取值划分出来的子集
子集中不再含该特征任意取值
'''
def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


# ---------------选择最好的划分特征-------------------------------------------
'''
把按每个特征划分得到的多个矩阵（该特征有几个值，就会划分出多少个矩阵）计算其香农熵总和。
得到的就是按该标签分类后的香农熵，找出最小香农熵的
'''
def choose_best_feature_to_split(dataset):
    num_feat = len(dataset[0]) - 1
    base_entropy = calc_Shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_feat):
        featlist = [example[i] for example in dataset]                  #把该特征列全部取出
        unique_vals = set(featlist)                                     #去重
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset,i,value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob * calc_Shannon_entropy(sub_dataset)     #计算划分后数据的香农熵
        info_gain = base_entropy - new_entropy                          #划分后，数据变有序，香农熵减少。减少的部分为信息增益
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# -----------------分类列表数目统计----------------------------------------------
'''
获取出现次数最多的分类标签
在当使用完所有特征，仍不能划分成唯一标签的情况
'''
def majority_class(class_list):
    class_count = {}
    for vote in class_list:
        vote = vote[0]
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key = operator.itemgetter(1),reverse = True)
    return sorted_class_count[0][0]


# -----------------------创建树-------------------------------------------------------
'''
递归构建树
关键是语句是：
mytree = {best_feat:{]}
mytree[best_feat][value] = create_tree()
注意该函数之后，labels有删减。若要使用完整的labels，要重新创建
'''
def create_tree(dataset,labels):#先设置叶子点要return的值，然后准备好各参数，在每个val下进行递归
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):     #第一个停止条件：如果标签全部一样了（可能还有几个特征没用上），就返回该标签并停止
        return class_list[0]
    if len(dataset[0]) == 1:                                   #第二个停止条件：如果所有特征都用完了，但还没找到唯一分类标签（还有好几个不同分类），就返回数量最多的标签并停止
        return majority_class(dataset)

    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    del(labels[best_feat])
    mytree = {best_feat_label:{}}
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)

    for value in unique_vals:
        sublabels = labels
        subdata = split_dataset(dataset,best_feat,value)
        mytree[best_feat_label][value] = create_tree(subdata,sublabels)
    return mytree


# ----------------------获取树的叶子数---------------------------------------------
'''
若为字典，进入迭代
若不为字典，那肯定是叶子，那么叶子数加一
'''
def get_leaves_num(input_tree):
    leaves_num = 0
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    #遍历每个特征的取值
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leaves_num += get_leaves_num(second_dict[key])
        else:
            leaves_num += 1
    return leaves_num

# ----------------------获取树的层数-----------------------------------------------
def get_depth_num():
    leaves_num = 0
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    #遍历每个特征的取值
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_depth_num(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

    

# --------------------使用决策树进行分类------------------------------------------------
'''
使用构建好的决策树进行分类
'''
def classify(input_tree,labels,test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]

    feat_index = labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key],labels,test_vec)
            else:
                class_label = second_dict[key]
    return class_label


#-------------------------便利函数-------------------------------------------------------
def classify_Tree(filename,test_vec):
    data,labels  = create_dataset(filename)
    tree = create_tree(data,labels)
    print tree
    data,labels  = create_dataset(filename)
    judgement = classify(tree,labels,test_vec)
    return judgement


###########################入口###########################################


a = classify_Tree('birds.csv',['1','1'])
print a
