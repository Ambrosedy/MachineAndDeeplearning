#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test
#-------------------------------------------------------------------------------

import sys
path = 'D:\\Ambroseslife\\Study\\jar\\libsvm-3.22\\python'
sys.path.append(path)
from svmutil import *
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import random


def getData():
    data = []
    
    feature1 = []
    with open('feature1.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature1.append(row)

    feature1 = np.array(feature1)
    feature1 = feature1[:,1:]
    data.append(feature1)

    feature2 = []
    with open('feature2.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature2.append(row)

    feature2 = np.array(feature2)
    feature2 = feature2[:,1:]
    data.append(feature2)

    feature3 = []
    with open('feature3.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature3.append(row)

    feature3 = np.array(feature3)
    feature3 = feature3[:,1:]
    data.append(feature3)

    feature4 = []
    with open('feature4.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature4.append(row)

    feature4 = np.array(feature4)
    feature4 = feature4[:,1:]
    data.append(feature4)

    feature5 = []
    with open('feature5.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature5.append(row)

    feature5 = np.array(feature5)
    feature5 = feature5[:,1:]
    data.append(feature5)

    feature6 = []
    with open('feature6.txt', 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            row = []
            row.append(int(linearr[0]))
            linearr2 = linearr[1].split(',')
            for i in range(len(linearr2)):
                row.append(int(linearr2[i]))
            feature6.append(row)

    feature6 = np.array(feature6)
    feature6 = feature6[:,1:]
    data.append(feature6)
    return data

def feat(feature1,feature2):
    #feature = np.c_[3*feature1, 4*feature2, 5*feature3, feature4, 4*feature5, 100*feature6]
    #feature = np.c_[feature1, 1.3*feature3]
    feature = np.c_[feature1,feature2]
    
    #feature = np.c_[feature1, feature2, 1.5*feature3, feature4, feature5, feature6]
    fea = feature.tolist()
    return fea
#特征拼接使用
def handle(W,feature):
    
    feature = np.c_[W[0]*feature[0],W[1]*feature[1],W[2]*feature[2],W[3]*feature[3],W[4]*feature[4],W[5]*feature[5]]
    
    return feature.tolist()

classification_num = 13 #26
allclass = [10, 11, 12, 20, 22, 25, 26, 28, 30, 31, 32, 33, 34]# 110 111 112 120 122 125 126 128 130 131 132 133 134];   #手动校正部分将黑底白字和白底黑字加起来
indexInfo = ['京', '渝', '鄂', '0',  '2', '5', '6', '8', 'A', 'B', 'C', 'D', 'Q']# '京' '渝' '鄂' '0'  '2' '5' '6' '8' 'A' 'B' 'C' 'D' 'Q'];
def RWP(feat):
     #train_num = 800
    selection_index = [];
    with open('selection_index1.txt', 'r') as fp:
        line = fp.readline()
        linearr = line.strip().split(' ')
        for i in range(len(linearr)):
            selection_index.append(int(linearr[i]))

    rno = 0
    clast = [] #训练样本集的标签
    clasv = [] #测试样本集的标签
    featt = [] #训练样本集的特征
    featv = [] #测试样本集的特征
    namet = [] #训练样本集的名称
    namev = [] #测试样本集的名称
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                if selection_index[rno] == 1:
                    clast.append(int(linearr[1]))
                    namet.append(linearr[2])
                    featt.append(feat[rno])
                else:
                    clasv.append(int(linearr[1]))
                    namev.append(linearr[2])
                    featv.append(feat[rno])
                rno += 1
    #param = svm_parameter('-t 1 -d %d -g 0.1 -r 0')
    #print(o)
    #model = svm_train(clast, featt, '-t 1 -d %d -g 0.1 -r 0'%o)
    model = svm_train(clast, featt, '-t 1 -d 1 -g 0.1 -r 0')
    pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(clasv, featv, model)
    return ACC,model
    
    
def CAP(feat):
    
    #train_num = 800
    selection_index = [];
    with open('selection_index1.txt', 'r') as fp:
        line = fp.readline()
        linearr = line.strip().split(' ')
        for i in range(len(linearr)):
            selection_index.append(int(linearr[i]))

    rno = 0
    clast = [] #训练样本集的标签
    clasv = [] #测试样本集的标签
    featt = [] #训练样本集的特征
    featv = [] #测试样本集的特征
    namet = [] #训练样本集的名称
    namev = [] #测试样本集的名称
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                if selection_index[rno] == 1:
                    clast.append(int(linearr[1]))
                    namet.append(linearr[2])
                    featt.append(feat[rno])
                else:
                    clasv.append(int(linearr[1]))
                    namev.append(linearr[2])
                    featv.append(feat[rno])
                rno += 1
    #param = svm_parameter('-t 1 -d %d -g 0.1 -r 0')
    #print(o)
    #model = svm_train(clast, featt, '-t 1 -d %d -g 0.1 -r 0'%o)
    model = svm_train(clast, featt, '-t 1 -d 1 -g 0.1 -r 0')
    pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(clasv, featv, model)
    

    print('ACC = ' + str(ACC))
    err_idx = []
    for i in range(len(clasv)):
        if clasv[i] != pred_labels[i]:
            err_idx.append(i)

    for i in range(len(err_idx)):
        e = allclass.index(clasv[err_idx[i]])
        f = allclass.index(pred_labels[err_idx[i]])
        print(indexInfo[e] + '被误识为' + indexInfo[f])

        imgpath = 'Char_Image/' + namev[err_idx[i]]
        img = io.imread(imgpath)
        plt.figure(i)
        plt.imshow(img)
    plt.show()
    return ACC
'''
    

def drawa():
    fig = plt.figure(figsize=(10,10))
    x = np.linspace(-9,9,19,dtype=int)
    y=[]
    for i in x:
        a = CAP(fea,i)
        y.append(a)
    plt.xlabel("coef0")
    plt.ylabel("ACC")
    plt.plot(x,y)
    plt.show
'''
def splite():
    lst = random.sample(range(0,1000),1000)
    ret = [x<800 for x in lst]
    
    with open('selection_index1.txt','w') as fp:
        for x in ret:
            if x:
                fp.write('1 ')
            else:
                fp.write('0 ')

def drawb():
    x = np.linspace(0,8,10,endpoint=False)
    y=[]
    for i in x:
        fea = feat(li[5]*2.5,li[2]*4)
        y.append(calMeanAcc(fea))
    plt.xlabel("w6")
    plt.ylabel("ACC")
    plt.plot(x,y)
    plt.show
#计算平均准确率
def calMeanAcc(fea):
    acc = 0
    i=0
    while(i<100):
        ac,model = RWP(fea)
        #if ac!=100: break
        acc+=ac    #读取随机数，从新划分训练集测试集、训练模型、预测、准确率求和
        splite()         #划分随机数800训练，200测试
        i+=1
    return acc/100,model

'''


def grad(w,a,feature):
    k = w
    acc,m = cal(k,feature)
    for i in range(6):
        temp = k
        #count = 0
        #m=temp[i]
        while(True):
            
            temp[i] += a
            c,m = cal(temp,feature)
            if c == 100:
                k = temp
                return k,m
            elif c>acc:
                acc = c
                k = temp
                #m=temp[i]
            elif c<=acc:
                break
          
    return k,m

def cleara():
    clast.clear()
    clasv.clear()
    featt.clear()
    featv.clear()
    namet.clear()
    namev.clear()
#根据W计算准确率
def cal(w,feature):
    clast.clear()
    clasv.clear()
    featt.clear()
    featv.clear()
    namet.clear()
    namev.clear()
    
    feat = handle(w,feature)
    train_test_getData(feat)
    model = svm_train(clast, featt, '-t 1 -d 1 -g 0.1 -r 0')
    pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(clasv, featv, model)
  

    return ACC,model

def draw():
    
    plt.figure(figsize=(10,10))
    plt.plot(pred_labels,linestyle='--',color = 'blue')
    plt.plot(clasv,linestyle='-',color = 'red')
    plt.show()
    plt.figure(figsize=(10,10))
    plt.scatter(clasv,pred_labels)
    plt.show()
    
def draw1(k,feature):
    f = plt.figure(figsize=(10,10))
    x = np.linspace(0,2,30)
    w = np.array([0.,0.,0.,0.,0.,0.])
    y = []
    for i in x:
        w[k] = i
        
        y.append(cal(w,feature))
    plt.plot(x,y)
    plt.show()

def cleara():
    clast.clear()
    clasv.clear()
    featt.clear()
    featv.clear()
    namet.clear()
    namev.clear()
#根据W计算准确率
def cal(w,feature):
    clast.clear()
    clasv.clear()
    featt.clear()
    featv.clear()
    namet.clear()
    namev.clear()
    
    feat = handle(w,feature)
    train_test_getData(feat)
    model = svm_train(clast, featt, '-t 1 -d 1 -g 0.1 -r 0')
    pred_labels, (ACC, MSE, SCC), pred_values = svm_predict(clasv, featv, model)
  

    return ACC,model

def draw():
    
    plt.figure(figsize=(10,10))
    plt.plot(pred_labels,linestyle='--',color = 'blue')
    plt.plot(clasv,linestyle='-',color = 'red')
    plt.show()
    plt.figure(figsize=(10,10))
    plt.scatter(clasv,pred_labels)
    plt.show()
    
def draw1(k,feature):
    f = plt.figure(figsize=(10,10))
    x = np.linspace(0,2,30)
    w = np.array([0.,0.,0.,0.,0.,0.])
    y = []
    for i in x:
        w[k] = i
        
        y.append(cal(w,feature))
    plt.plot(x,y)
    plt.show()
'''
'''
def grad(w,a,feature):
    k = w
    acc = calMeanAcc(feature)
    for i in range(6):
        temp = k
        #count = 0
        #m=temp[i]
        while(True):
            
            temp[i] += a
            fea = handle(temp,li)
            c = calMeanAcc(fea)
            if c == 100:
                k = temp
                return k
            elif c>acc:
                acc = c
                k = temp
                #m=temp[i]
            elif c<=acc:
                break
          
    return k
'''
if __name__ == '__main__':
    li=getData()
   # fea = feat(li[1])
    #CAP(fea)
   # w = [3,4,10,1,4,100]
   # w=[0,0,2.2,0,0,2.4]
   # w=[0,0,0,0,0,1]
    w=[0,0,1.8,0,0,2.2]
    fea = handle(w,li)
'''   
    while(True):
        splite()
        acc,model = calMeanAcc(fea)
        if acc==100:
            break
       '''
    #grad(w,0.05,fea)
''' 
    x = np.linspace(1.5,3.0,20,endpoint=False)
    y=[]
    for i in x:
        w[2] = i
        fea = handle(w,li)
        acc,model = calMeanAcc(fea)
        y.append(acc)
    plt.plot(x,y)

    acc=0
    i=0
    while(i<100):
        acc+=CAP(fea)
        splite()
        i+=1
    print(acc/100)
'''
   
''' while(i<100):
        i+=1
        if(acc==100):
            splite()
            acc = test()

        else:
            print(i)
            print("******************")
            break'''
