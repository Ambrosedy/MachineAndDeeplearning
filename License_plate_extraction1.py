#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[4]:


image0 = cv2.imread("1.jpg")
'''cv2.imshow("img",image0)
cv2.waitKey()                      '''           #不写就会未响应


# In[7]:


gimage = cv2.GaussianBlur(image0,(5,5),0)
'''cv2.imshow("img1",gimage)
cv2.imshow("img",image0)
cv2.waitKey()'''


# In[8]:


gray=cv2.cvtColor(gimage,cv2.COLOR_BGR2GRAY)
'''cv2.imshow("img3",gray)
cv2.waitKey()'''


# In[9]:


Sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
# Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
# absY = cv2.convertScaleAbs(Sobel_y)
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
gray = absX                                                  #边缘检测


# In[10]:


ret,gray = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
'''cv2.imshow("img3",gray)
cv2.waitKey()'''


# In[17]:


kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(20,5))
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernelx)
cv2.imshow("img3",gray)
cv2.waitKey()


# In[28]:


kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
image = gray
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)

image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)


# In[ ]:





# In[29]:


image = cv2.medianBlur(image,25)
cv2.imshow("img3",image)
cv2.waitKey()


# In[14]:


contours, w1 = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
# 绘制轮廓
image = cv2.drawContours(image0, contours, -1, (0, 0, 255), 3)
cv2.imshow('image', image)
cv2.waitKey()


# In[30]:


for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2):
        image1 = image0[y:y + height, x:x + weight]
        cv2.imshow('image', image1)
        cv2.waitKey()


# In[ ]:




