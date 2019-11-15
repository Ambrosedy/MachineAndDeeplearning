# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:36:38 2019

@author: 汐19990223
"""

from __future__ import print_function  
from skimage import io, filters
import numpy as np
np.random.seed(4324)  # for reproducibility  用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同
  
from PIL import Image  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD  
from keras.utils import np_utils
from keras import backend as K


# There are 40 different classes  
nb_classes = 13  # 40个类别
epochs = 30 # 进行40轮次训  50可以 45可以 43、42可以
batch_size = 13# 每次迭代训练使用40个样本     #20可以 30可以  35可以 38可以
  
# input image dimensions  
img_rows, img_cols = 47, 92  
# number of convolutional filters to use  
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
# size of pooling area for max pooling  
nb_pool = 2  
# convolution kernel size  
nb_conv = 3  # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
#加载数据
def newlabel(label):
    if label<20:
        if label==10:
            return 1
        elif label==11:
            return 3
        else:return 4
    elif label<30:
        return label%20
    elif label==30:
        return 7
    elif label>30:
        return label%30+8

def load_data():
    url1 = []
    train = np.empty((800,4324))
    train_label = np.empty(800)                            #概率计算
    test = np.empty((100,4324))
    test_label = np.empty(100)
    valid = np.empty((100,4324))
    valid_label = np.empty(100)
    number = np.empty((1000,4324))
    label = []
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
           if line[0] != '#':
               linearr = line.strip().split('\t')
               url1.append(linearr[2])                                #存储图片路径
               label.append(newlabel(int(linearr[1])))                               #标签
    
    train_index = np.random.choice(len(number),np.int(len(number)*0.8),replace=False)
    other = np.array(list(set(range(len(number)))-set(train_index)))
    valid_index = np.random.choice(other,np.int(len(other)*0.5),replace=False)
    test_index = np.array(list(set(other)-set(valid_index)))
    s=0
    d=0
    f=0
    for i in range(1000):
        img = io.imread('Char_Image_Binary/' + url1[i])
        t = filters.threshold_otsu(img)
        A = (img>t)*1                                              #二值化
        number[i] = np.ndarray.flatten(A)                        #将A拉直
        
        
        if i in train_index:
            train[s] = number[i]
            train_label[s] = label[i]
            s+=1
        elif i in test_index:
            test[d] = number[i]
            test_label[d] = label[i]
            d+=1
        else:
            valid[f] = number[i]
            valid_label[f] = label[i]
            f+=1
            
    rval = [(train, train_label), (valid, valid_label), (test, test_label)] 
    
    return rval
        
def set_model(lr=0.005,decay=1e-6,momentum=0.9):   #lr学习率（0.05、0.04）、decay= momentum=冲量
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        #滤波时取中间值，2x2无中间值
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape = (1, img_rows, img_cols)))  #5 : 特征数：人脸，经验根据
    else:
        model.add(Conv2D(5, kernel_size=(3, 3), input_shape = (img_rows, img_cols, 1)))
        #根据图片特征样式可以调整这个卷积核的样式
    model.add(Activation('elu')) #sigmoid，relu，tanh，elu   数据顺序
    model.add(MaxPooling2D(pool_size=(2, 2)))  #
    model.add(Conv2D(10, kernel_size=(3, 3)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Dropout(0.2))
    model.add(Flatten())      
    model.add(Dense(128)) #Full connection  
    model.add(Activation('elu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))  
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)#keras.losses.binary_crossentropy  'categorical_crossentropy'
    return model          
        
    
    
def train_model(model,X_train, Y_train, X_val, Y_val):  
    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,  
          verbose=1, validation_data=(X_val, Y_val))  
    model.save_weights('model_weights.h5', overwrite=True)  
    return model  
  
def test_model(model,X,Y):  
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X, Y, verbose=0)
    return score      
    
    
if __name__ == '__main__':
    #加载数据，区分训练集、测试集、
     (X_train, y_train), (X_val, y_val),(X_test, y_test) = load_data()
     
     if K.image_data_format() == 'channels_first':   #判断通道：意为通道数在第一个
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)  
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
        input_shape = (1, img_rows, img_cols)  #1 是通道数
     else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)  
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
        input_shape = (img_rows, img_cols, 1) # 1 为图像像素深度
    
     print('X_train shape:', X_train.shape)
     print(X_train.shape[0], 'train samples') 
     print(X_val.shape[0], 'validate samples')  
     print(X_test.shape[0], 'test samples')
  
    # convert class vectors to binary class matrices  
     Y_train = np_utils.to_categorical(y_train, nb_classes)    #将整型的类别标签转为onehot编码。y为int数组，num_classes为标签类别总数，大于max(y)（标签从0开始的）
     Y_val = np_utils.to_categorical(y_val, nb_classes)
     Y_test = np_utils.to_categorical(y_test, nb_classes)
     
     model = set_model()    #add  在纸上画一下
     train_model(model, X_train, Y_train, X_val, Y_val)   
     score = test_model(model, X_test, Y_test)  
  
     model.load_weights('model_weights.h5')  
     classes = model.predict_classes(X_test, verbose=0)  
     test_accuracy = np.mean(np.equal(y_test, classes))  
     print("accuarcy:", test_accuracy)
     for i in range(0,40):
        if y_test[i] != classes[i]:
            print(y_test[i], '被错误分成', classes[i]);
    