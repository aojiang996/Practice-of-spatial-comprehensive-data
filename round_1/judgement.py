# coding=utf-8
'''
Created on 2020-6-13
@author: jiangao
Project: 图像评价标准
'''
import numpy as np
import cv2
import math
import time

def standard_deviation(image):
    """
    计算标准差,标准差越大,分辨率越高
    """
    data = np.std(image, ddof = 1)
    return data

def information_entropy(image):
    """
    计算信息熵,信息熵越大,图像越复杂
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h , w = image.shape
    pixels = h * w
    num = np.zeros(256)
    for i in image:
        for j in i:
            num[j] = num[j] + 1
    num = num / pixels
    entropy = 0
    for i in range(len(num)):
        if num[i] != 0 :
            entropy -= num[i]*math.log2(num[i])
    return entropy

def spatial_frequency(image):
    """
    计算空间频率,空间频率越大,图像越清晰
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float)
    h , w = image.shape
    pixels = h * w
    row = 0.0
    col = 0.0
    for i in image:
        for j , k in zip(i,i[1:]):
            row += ( j - k )**2
    
    for i , k in zip(image,image[1:]):
        for j in range(w):
            col += ( i[j] - k[j] )**2

    row = row / pixels
    col = col / pixels
    return math.sqrt(row + col)

def average_gradient(image):
    """
    计算平均梯度，清晰度评价准则值越大表示图像越清晰，反之，图像越模糊
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float)
    h , w = image.shape
    pixels = h * w
    sobelx=cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    sums = []
    for i , j in zip(sobelx,sobely):
        for k , l in zip(i,j):
            sums.append(math.sqrt(k**2 + l**2))
    average_sums = sum(sums) / pixels
    return average_sums

def readHSV():
    """
    打开HSV融合图像
    """
    path = r'C:\Users\64908\Desktop\HSV.png'
    image = cv2.imread(path)
    return image

def readBrovey():
    """
    打开Brovey融合图像
    """
    path = r'C:\Users\64908\Desktop\Brovey.png'
    image = cv2.imread(path)
    return image

def readIHS():
    """
    打开IHS融合图像
    """
    path = r'C:\Users\64908\Desktop\IHS.png'
    image = cv2.imread(path)
    return image

if __name__ == "__main__":
    brovey = readHSV()
    br = cv2.imread(r'C:\Users\64908\Desktop\HSV_classical.png')
    print('{:{}<8}{:{}<6}{:9.4f}{:{}>9}{:9.4f}'.format('HSV标准差：',chr(12288),'代码效果：',chr(12288),standard_deviation(brovey),'envi效果：',chr(12288),standard_deviation(br)))
    print('{:{}<8}{:{}<6}{:9.4f}{:{}>9}{:9.4f}'.format('HSV信息熵：',chr(12288),'代码效果：',chr(12288),information_entropy(brovey),'envi效果：',chr(12288),information_entropy(br)))
    print('{:{}<8}{:{}<6}{:9.4f}{:{}>9}{:9.4f}'.format('HSV空间频率：',chr(12288),'代码效果：',chr(12288),spatial_frequency(brovey),'envi效果：',chr(12288),spatial_frequency(br)))
    print('{:{}<8}{:{}<6}{:9.4f}{:{}>9}{:9.4f}'.format('HSV平均梯度：',chr(12288),'代码效果：',chr(12288),average_gradient(brovey),'envi效果：',chr(12288),average_gradient(br)))
