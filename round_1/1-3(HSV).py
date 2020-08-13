# coding=utf-8
'''
Created on 2020-6-1
@author: jiangao
Project: HSV图像融合方法
'''
import numpy as np
import cv2
import scipy.misc as smi
from PIL import Image
from osgeo import gdal

def HSV(data_low,data_high):
    """
    使用opencv将色彩空间RGB转为HSV,H色调,S饱和度,V明度
    输入：图像三维数组
    返回：可绘出图像的utf-8格式的三维数组
    """
    h_low , s_low , v_low = cv2.split(
        cv2.cvtColor(
            data_low,cv2.COLOR_BGR2HSV
        )
    )
    h_high , s_high , v_high = cv2.split(
        cv2.cvtColor(
            data_high,cv2.COLOR_BGR2HSV
        )
    )
    #用高分影像的V替换低分影像的V
    HSV = cv2.merge([h_low,s_low,v_high])
    RGB = Image.fromarray(
        cv2.cvtColor(
            HSV,cv2.COLOR_HSV2RGB
        )
    )
    return RGB
def main(path_low,path_high):
    data_low = cv2.imread(path_low)
    data_high = cv2.imread(path_high)
    h,w = data_high.shape[:2]
    data_low = cv2.resize(data_low,(h,w),interpolation=cv2.INTER_CUBIC)#重采样
    RGB = HSV(data_low,data_high)
    RGB.save(r'C:\Users\64908\Desktop\HSV2.png','png')
if __name__ == "__main__":
    path_low = r'C:\Users\64908\Desktop\RGB.tif'
    path_high = r'C:\Users\64908\Desktop\Band8.tif'
    main(path_low,path_high)