# coding=utf-8
'''
Created on 2020-6-1
@author: jiangao
Project: PCA图像融合方法
'''
import numpy as np
import cv2
import scipy.misc as smi
from PIL import Image
from osgeo import gdal

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
def PCA(data_low,data_high):
    """
    主成分图像融合
    输入：numpyArray格式的三维数组
    返回：可绘出图像的utf-8格式的三维数组
    """
    band , col , row = data_high.shape
    pixels = col * row
    #2002*2002——>pixels*3
    data_low = data_low.reshape((pixels,3))
    data_high = data_high.reshape((pixels,3))
    templow = data_low
    temphigh = data_high
    #按列取平均值(pixels*3——>1*3)
    data_low_mean = np.mean(data_low,0)
    data_high_mean = np.mean(data_low,0)
    #每列减去对应的平均值
    data_low = data_low - np.tile(
        data_low_mean,(pixels,1)
    )
    data_high = data_high - np.tile(
        data_high_mean , (pixels,1)
    )
    #求协方差矩阵
    correlation = (np.matrix(data_low).T * np.matrix(data_low)) / pixels
    highcorrelation = (np.matrix(data_high).T * np.matrix(data_high)) / pixels
    #求特征向量与特征值
    low_value , low_vector = np.linalg.eig(correlation)
    high_value , high_vector = np.linalg.eig(highcorrelation)
    #将特征向量左右对调
    low_vector = np.fliplr(low_vector)
    high_vector = np.fliplr(high_vector)
    #PCA正变换
    RGB = np.array(
        np.matrix(templow) * np.matrix(low_vector)
    )
    band8 = np.array(
        np.matrix(temphigh) * np.matrix(high_vector)
    )
    #用高分影像第一主分量代替低分影像第一主分量
    RGB[:,0] = band8[:,0]
    #将合成的数据进行PCA逆变换，获得高分辫率的多光谱融合图像
    new_templow = np.array(
        np.matrix(RGB) * np.linalg.inv(low_vector)
    )
    new_RGB = new_templow.reshape((band,col,row))
    min_val = np.min(new_RGB.ravel())
    max_val = np.max(new_RGB.ravel())
    RGB = np.uint8(
        (new_RGB.astype(np.float) - min_val) / (max_val - min_val) * 255
    )
    RGB = Image.fromarray(
        cv2.merge(
            [RGB[0],RGB[1],RGB[2]]
        )
    )
    return RGB 
def main(path_low,path_high):
    data_low = gdal_open(path_low)
    data_high = gdal_open(path_high)
    data_low = imresize(data_low,data_high)
    RGB = PCA(data_low,data_high)
    RGB.save(r"D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\PCA3.png",'png')
if __name__ == "__main__":
    path_low = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\RGB.tif'
    path_high = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\Band8.tif'
    main(path_low,path_high)