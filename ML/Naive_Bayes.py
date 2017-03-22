# -*- coding:utf-8 -*-

'''
训练文本必须是utf-8格式，不能是unicode
docs_list是所有已经分好词的训练文本。每条文本为一个列表，总文本是大列表。
class_list是所有训练文本的分类
'''

import numpy as np
import jieba


###--------------根据关键词分类（在本程序中不会用到）-------------------------------------------------------
##'''
##如果是侮辱性文本，标记为1，否则为0
##若有多种文本类型，更改添加判断处即可
##'''
##def get_class_list(docs_list,abusive_words):
##    class_list = [0]*len(docs_list)
##    for doc in docs_list:
##        for word in doc:
##            if word in abusive_words:
##                class_list[docs_list.index(doc)] = 1
##                break
##    return class_list

#------------------  获取数据------------------------------------------------------
'''
从文本中获取切分好的各条文本和分类列表
注意源文件编码格式为utf-8
'''
def get_docs_and_class_list(filename):
    fr = open(filename)
    content = fr.read()
    fr.close()
    content = content.split('\n')
    docs  = [' '.join(jieba.cut(i[1:])).split(' ') for i in content]
    class_list = []
    for i in content:
        class_list.append(int(i[0]))
    return docs, class_list


# #----------------------导入词库-----------------------------------------------------------
# def create_vocabulary_list(filename):
#     fr = open(filename)
#     content = fr.read()
#     fr.close()
#     content = re.sub(r'[\d]','',content)
#     vocabulary_list = content.split('\n')
#     return vocabulary_list


# -------------------创建词汇表-------------------------------------------------------
'''
创建出一个包含所有单词的列表（词汇表）
'''
def create_vocabulary_list(docs_list):
   vocabulary_set = set([])
   for doc in docs_list:
       vocabulary_set = vocabulary_set | set(doc)
   vocabulary_list = list(vocabulary_set)
   return vocabulary_list


#---------------把每个文本映射到词汇表中----------------------------------------------

def doc2vec(doc,vocabulary_list):
    doc_vec = [0]*len(vocabulary_list)
    for word in doc:
        if word in vocabulary_list:
            doc_vec[vocabulary_list.index(word)] += 1
        else:
            print 'The word %s is not in my Vocabulary' % word
    return doc_vec


#------------------把文本列表映射到词汇表中---------------------------------------

def get_docs_mat(docs_list,vocabulary_list):
    docs_mat = []
    for doc in docs_list:
        docs_mat.append(doc2vec(doc,vocabulary_list))
    return docs_mat
    
#--------------统计各词在不同类型文本中出现的频率，即  P(词|分类)  --------------------------------------------
'''
若有多种分类，更改判断处及p_class即可
'''
def train_NB(docs_mat,class_list):
    docs_num = len(docs_mat)
     
    words_num = len(docs_mat[0])                        #所有单词数
    
    p_class0 = class_list.count(0)/float(docs_num)
    p_class1 = class_list.count(1)/float(docs_num)      #侮辱性文本的概率
    p_class2 = class_list.count(2)/float(docs_num)
    
    p0_num = np.ones(words_num)  #拉普拉斯平滑,加lambda。相当于事先给定一个均匀先验分布
    p1_num = np.ones(words_num)
    p2_num = np.ones(words_num)

    p0_denom = 3.0#拉普拉斯平滑,加K*lambda
    p1_denom = 3.0
    p2_denom = 3.0

    for i in range(docs_num):
        if class_list[i] == 0:              #如果分类为0，统计各词在其中出现的频率
            p0_num += docs_mat[i]#count
            p0_denom += sum(docs_mat[i])#该分类下所有单词数
        if class_list[i] == 1:              #如果分类是1，统计各词在其中出现的频率
            p1_num += docs_mat[i]
            p1_denom += sum(docs_mat[i])
        if class_list[i] == 2:              #如果分类是2，统计各词在其中出现的频率
            p2_num += docs_mat[i]
            p2_denom += sum(docs_mat[i])
            
    p_words_in_c0 = np.log(p0_num/p0_denom) #将概率进行对数变换，避免０的相乘，同时应用独立性时，相乘变相加
    p_words_in_c1 = np.log(p1_num/p1_denom)
    p_words_in_c2 = np.log(p2_num/p2_denom)
    return p_words_in_c0,p_words_in_c1,p_words_in_c2,p_class0,p_class1,p_class2


#-------------------分类-------------------------------------------------------------
def classify(input_doc_vec,p_words_in_c0,p_words_in_c1,p_words_in_c2,p_class0,p_class1,p_class2):
    p0 = sum(input_doc_vec * p_words_in_c0)+np.log(p_class0)#由于进行了对数变换，因此相乘变相加
    p1 = sum(input_doc_vec * p_words_in_c1)+np.log(p_class1)
    p2 = sum(input_doc_vec * p_words_in_c2)+np.log(p_class2)
    max_p = max(p0,p1,p2)
    if max_p == p0:
        return 0
    if max_p == p1:
        return 1
    if max_p == p2:
        return 2


#-------------------便利函数---------------------------------------------------------
def classify_NB(docs_list,class_list,input_doc):
#    docs_list = [[word.lower() for word in doc] for doc in docs_list]
    vocabulary_list = create_vocabulary_list(docs_list)   #获得词汇表，也可以自己创建
    docs_mat = get_docs_mat(docs_list,vocabulary_list)    #把文本转化为向量
    p0,p1,p2,p_c0,p_c1,p_c2 = train_NB(docs_mat,class_list)       #获得单词在不同文本中出现的概率
    input_doc = ' '.join(jieba.cut(input_doc)).split(' ')
#    input_doc = [word.lower() for word in input_doc]
    input_doc_vec = doc2vec(input_doc,vocabulary_list)
    judgement = classify(input_doc_vec,p0,p1,p2,p_c0,p_c1,p_c2)
    return judgement


#######################入口##############################################################

##docs_list = [['My','dog','has','flea','problems','help','please'],
##               ['maybe','not','take','him','to','dog','park','stupid'],
##               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
##               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
##               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
##               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
##class_list = [0,1,0,1,0,1]
##
##
##a = classify_NB(docs_list,class_list,'love my Dalmation')
##print a
input_words = raw_input('请输入微博：  ')
docs, class_list = get_docs_and_class_list('weibo(Male_HighSchool).txt')
judgement = classify_NB(docs, class_list,input_words)

if judgement == 0:
   print '文本： ' + input_words + '\n' + '情绪:  ' + '伤心'
if judgement == 1:
   print '文本： ' + input_words + '\n' + '情绪:  ' + '平和'
if judgement == 2:
   print '文本： ' + input_words + '\n' + '情绪:  ' + '开心'









