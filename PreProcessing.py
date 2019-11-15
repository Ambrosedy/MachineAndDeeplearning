#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        PreProcessing
#-------------------------------------------------------------------------------

import numpy as np
from skimage import io, color, filters, morphology

def PreProcessing():
    input3 = []
    with open('D:/code/pycode/cloudlane/Char_Index.txt', 'r') as fp:                            
        #文件内是文件名
        for line in fp:
            if line[0] != '#':
                linearr = line.strip().split('\t')
                input3.append(linearr[2])

    bl = 47
    for k in range(1000):
        A = io.imread('D:/code/pycode/cloudlane/Char_Image/' + input3[k], as_gray = True)         
        #读取图片
        #io.imshow(A)
        #io.show()
        A = color.rgb2gray(A)
        t = filters.threshold_otsu(A)                                                                  
        
        #阈值、二值化
        B = (A > t) * 1.0
        a = B.shape[0]
        b = B.shape[1]
        
        B = np.zeros((a,b))   
        
        for i in range(0, a, bl):
            for j in range(0, b, bl):
                mid = A[i:min(i+bl, a), j:min(j+bl, b)]             #图像上取局部
                t = filters.threshold_otsu(mid)                     #得到阈值
                mid_t = (mid > t) * 1.0                             #true 1,false 0
              #  print(mid_t)
              #  print(B[i:min(i+bl, a), j:min(j+bl, b)])
                B[i:min(i+bl, a), j:min(j+bl, b)] = mid_t

        B = morphology.opening(B, morphology.disk(2))               #腐蚀，如果点比disk小，腐蚀掉；morphology形态学
        
      
        if B[0, 0] + B[0, b-1] + B[a-1, 0] + B[a-1, b-1] >= 2:                     
            #如果背景是白色   
            for i in range(a):
                for j in range(b):
                    B[i, j] = 1 - B[i, j]                                          
                    #黑白颠倒
         
        io.imsave('D:/code/pycode/cloudlane/Char_Image_Binary/' + input3[k], B)
    pass

if __name__ == '__main__':
    PreProcessing()




