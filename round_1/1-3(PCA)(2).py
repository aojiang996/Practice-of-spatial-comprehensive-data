# coding=utf-8
'''
Created on 2020-6-1
@author: jiangao
Project: PCA图像融合方法
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
def PCA(data_low,data_high):
    """
    主成分图像融合
    输入：numpyArray格式的三维数组
    返回：可绘出图像的utf-8格式的三维数组
    """
    band , h , w = data_high.shape
    pixels = h * w
    data_low = data_low.reshape((pixels,3))
    data_high = data_high.reshape((pixels,3))
    #按列取均值
    data_low_mean = np.array(
        np.matrix(
            np.mean(data_low,0)
        )
    )
    data_high_mean = np.array(
        np.matrix(
            np.mean(data_high,0)
        )
    )
    #计算差值
    data_low_reduce = data_low - np.tile(data_low_mean,(pixels,1))
    data_high_reduce = data_high - np.tile(data_high_mean,(pixels,1))
    #计算协方差
    data_low_reduce_T = np.array(
        np.matrix(
            data_low_reduce
        ).T
    )
    data_high_reduce_T = np.array(
        np.matrix(
            data_high_reduce
        ).T
    )
    data_low_covariance = np.dot(data_low_reduce_T, data_low_reduce) / (pixels)
    data_high_covarience = np.dot(data_high_reduce_T , data_high_reduce) / (pixels)
    data_low_covariance = np.cov(data_low[0])
    data_high_covariance = np.cov(data_high[0])
    #获取特征值和特征向量
    low_value , low_vector = np.linalg.eig(data_low_covariance)
    high_value , high_vector = np.linalg.eig(data_high_covariance)
    #PCA正变换
    data_low_PCA = np.array(
        np.matrix(
            np.dot(
                np.matrix(data_low) , np.fliplr(np.matrix(low_vector))
            )
        )
     )
    data_high_PCA = np.array(
        np.matrix(
            np.dot(
                np.matrix(data_high) , np.fliplr(np.matrix(high_vector))
            )
        )
     )
    #用高分影像的第一波段替换低分影像的第一波段
    data_low_PCA[:,0] = data_high_PCA[:,0]
    #PCA逆变换
    data_low_PCA_inv = np.array(
        np.matrix(
            np.dot(
                np.matrix(data_low_PCA) , np.matrix(low_vector)
            )
        )
    )
    data_high_PCA_inv = np.array(
        np.matrix(
            np.dot(
                np.matrix(data_high_PCA) , np.matrix(high_vector)
            )
        )
    )
    band8 = data_high_PCA_inv.reshape((band,h,w))
    RGB = data_low_PCA_inv.reshape((band,h,w))
    min_val = np.min(band8.ravel())
    max_val = np.max(band8.ravel())
    band8 = np.uint8((band8.astype(np.float) - min_val) / (max_val - min_val) * 255)
    band8 = Image.fromarray(cv2.merge([band8[0],band8[1],band8[2]]))
    min_val = np.min(RGB.ravel())
    max_val = np.max(RGB.ravel())
    RGB = np.uint8((RGB.astype(np.float) - min_val) / (max_val - min_val) * 255)
    RGB = Image.fromarray(cv2.merge([RGB[0],RGB[1],RGB[2]]))
    return RGB
def main(path_low,path_high):
    data_low = gdal_open(path_low)
    data_high = gdal_open(path_high)
    data_low = imresize(data_low,data_high)
    RGB = PCA(data_low,data_high)
    RGB.save(r"D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\PCA2.png",'png')
if __name__ == "__main__":
    path_low = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\RGB.tif'
    path_high = r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\Band8.tif'
    main(path_low,path_high)