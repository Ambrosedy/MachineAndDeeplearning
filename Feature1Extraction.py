#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Feature1Extraction
#-------------------------------------------------------------------------------

import numpy as np
from skimage import io, filters

def Feature1Extraction():
    input3 = []
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                input3.append(linearr[2])
    with open('feature1.txt','w+') as fp: #打开feature1.txt，用以储存特征1的数据
        for k in range(1000):
            A = io.imread('Char_Image_Binary/' + input3[k])
            t = filters.threshold_otsu(A) #取阈值
            B = (A > t) * 1 #二值化，B为二值化后的图像矩阵，每个元素的值为0或1
            a = B.shape[0]
            b = B.shape[1]
            C = np.zeros(a+b, dtype=np.int) #定义特征向量
            for i in range(a):
                for j in range(b):
                    if B[i,j] == 1:
                        C[i] += 1 #如果行中的元素为白色，则增加1，最终结果为每一行的白点数
                #C[i] /= b

            for j in range(b):
                for i in range(a):
                    if B[i, j] == 1:
                        C[a+j] += 1 #最终结果为每一列的白点数
                #C[a+j] /= a

            fp.write(str(k+1) + '\t')
            for i in range(a+b-1):
                fp.write(str(C[i]) + ',')
            if k < 999:
                fp.write(str(C[a+b-1]) + '\n')
            else:
                fp.write(str(C[a+b-1]))

    pass

if __name__ == '__main__':
    Feature1Extraction()
