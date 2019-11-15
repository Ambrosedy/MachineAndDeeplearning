#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        Feature3Extraction
#-------------------------------------------------------------------------------

import numpy as np
from skimage import io, filters

def Feature3Extraction():
    input3 = []
    with open('Char_Index.txt', 'r') as fp:
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                input3.append(linearr[2])
    with open('feature3.txt','w+') as fp: #打开feature3.txt，用以储存特征1的数据
        for k in range(1000):
            A = io.imread('Char_Image_Binary/' + input3[k])
            t = filters.threshold_otsu(A) #取阈值
            B = (A > t) * 1 #二值化，B为二值化后的图像矩阵，每个元素的值为0或1
            a = B.shape[0]
            b = B.shape[1]
            C = np.zeros(a*2+b*2, dtype=np.int) #定义特征向量
            for i in range(a):
                for j in range(b):
                    if B[i,j] == 1:
                        C[i] = j # 由左边起，遇到白点，则计算出字符与左边距离
                        break
                    if j == b-1 and B[i,j] == 0:
                        C[i] = b #若这一行不存在字符元素，距离为最大，取b
            for i in range(a):
                for j in range(b-1, -1, -1):
                    if B[i,j] == 1:
                        C[a+i] = b-1-j #由右边起，遇到白点，计算出字符与右边距离
                        break
                    if j == 0 and B[i,j] == 0:
                        C[a+i] = b #若这一行不存在字符元素，距离为最大，取b
            for j in range(b):
                for i in range(a):
                    if B[i,j] == 1:
                        C[a*2+j] = i # 由上边起，遇到白点，计算出字符与右边距离
                        break
                    if i == a-1 and B[i,j] == 0:
                        C[a*2+j] = a # 若这一列不存在字符元素，距离为最大，取a
            for j in range(b):
                for i in range(a-1, -1, -1):
                    if B[i,j] == 1:
                        C[a*2+b+j] = a-1-i #由下边起，遇到白点，计算出字符与右边距离
                        break
                    if i == 0 and B[i,j] == 0:
                        C[a*2+b+j] = a #若这一列不存在字符元素，距离为最大，取a

            fp.write(str(k+1) + '\t')
            for i in range(a*2+b*2-1):
                fp.write(str(C[i]) + ',')
            if k < 999:
                fp.write(str(C[a*2+b*2-1]) + '\n')
            else:
                fp.write(str(C[a*2+b*2-1]))
    pass

if __name__ == '__main__':
    Feature3Extraction()
