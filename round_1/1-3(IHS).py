# coding=utf-8
'''
Created on 2020-6-1
@author: jiangao
Project: IHS图像融合方法
'''
import numpy as np
import cv2
import scipy.misc as smi
from osgeo import gdal
from PIL import Image

def gdal_open(path):
    """
    读取图像函数
    输入：图像路径
    返回：numpyArray格式的三维数组
    """
    data = gdal.Open(path)
    col = data.RasterXSize#读取图像长度
    row = data.RasterYSize#读取图像宽度
    data_array_r = data.GetRasterBand(1).ReadAsArray(0,0,col,row).astype(np.float)#读取图像第一波段并转换为数组
    data_array_g = data.GetRasterBand(2).ReadAsArray(0,0,col,row).astype(np.float)#读取图像第二波段并转换为数组
    data_array_b = data.GetRasterBand(3).ReadAsArray(0,0,col,row).astype(np.float)#读取图像第三波段并转换为数组
    data_array = np.array((data_array_r,data_array_g,data_array_b))
    return data_array
def imresize(data_low,data_high):
    """
    图像缩放函数
    输入：numpyArray格式的三维数组
    返回：numpyArray格式的三维数组
    """
    band , col , row = data_high.shape
    data = np.zeros(((band,col,row)))
    for i in range(0,band):
            data[i] = smi.imresize(data_low[i],(col,row))
    return data
def IHS(data_low,data_high):
    """
    基于IHS变换融合算法
    输入：numpyArray格式的三维数组
    返回：可绘出图像的utf-8格式的三维数组
    """
    A = [[1./3.,1./3.,1./3.],[-np.sqrt(2)/6.,-np.sqrt(2)/6.,2*np.sqrt(2)/6],[1./np.sqrt(2),-1./np.sqrt(2),0.]] 
    #RGB－>IHS正变换矩阵
    B = [[1.,-1./np.sqrt(2),1./np.sqrt(2)],[1.,-1./np.sqrt(2),-1./np.sqrt(2)],[1.,np.sqrt(2),0.]] 
    #IHS－>RGB逆变换矩阵
    A = np.matrix(A)
    B = np.matrix(B)

    band , w , h = data_high.shape
    pixels = w * h
    data_low = data_low.reshape(3,pixels)
    data_high = data_high.reshape(3,pixels)
    a1 = np.dot(A , np.matrix(data_high))#高分影像正变换
    a2 = np.dot(A , np.matrix(data_low))#低分影像正变换
    a2[0,:] = a1[0,:]#用高分影像第一波段替换低分影像第一波段
    RGB = np.array(np.dot(B , a2))#融合影像逆变换
    RGB = RGB.reshape((3,h,w))
    min_val = np.min(RGB.ravel())
    max_val = np.max(RGB.ravel())
    RGB = np.uint8((RGB.astype(np.float) - min_val) / (max_val - min_val) * 255)
    RGB = Image.fromarray(cv2.merge([RGB[0],RGB[1],RGB[2]]))
    return RGB
def main(path_low,path_high):
    data_low = gdal_open(path_low)
    data_high = gdal_open(path_high)
    data_low = imresize(data_low,data_high)
    RGB = IHS(data_low,data_high)
    RGB.save(r"D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\IHS.png",'png')
if __name__ == "__main__":
    path_low = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\RGB.tif'
    path_high = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\Band8.tif'
    main(path_low,path_high)
