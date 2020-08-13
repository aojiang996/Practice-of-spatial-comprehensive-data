# coding=utf-8
'''
Created on 2020-6-1
@author: jiangao
Project: HSV图像融合方法
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc as smi
import math
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
def rgb2hsv(r,g,b):
    """
    RGB转HSV格式
    输入：R，G，B numpyArray格式二维数组
    返回：H，S，V numpyArray格式二维数组
    """
    h = r.shape[0]
    w = r.shape[1]
    H = np.zeros((h, w))
    S = np.zeros((h, w))
    V = np.zeros((h, w))
    r, g, b = r/255.0, g/255.0, b/255.0

    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            dt = mx-mn
            #H
            if mx == mn:
                H[i, j] = 0
            elif mx == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = 60 * (g[i, j] - b[i, j]) / dt
                else:
                    H[i, j] = 60 * (g[i, j] - b[i, j]) / dt + 360
            elif mx == g[i, j]:
                H[i, j] = 60 * (b[i, j] - r[i, j]) / dt + 120
            elif mx == b[i, j]:
                H[i, j] = 60 * (r[i, j] - g[i, j]) / dt + 240
            #S
            if mx == 0:
                S[i, j] = 0
            else:
                S[i, j] = dt / mx 
            #V
            V[i, j] = mx
    return H, S, V
def hsv2rgb(h, s, v):
    """
    HSV转RGB格式
    输入：H，S，V numpyArray格式二维数组
    返回：R，G，B numpyArray格式二维数组
    """
    col , row = h.shape[0] , h.shape[1]
    hi = h / 60.0
    hif = hi.astype(np.int) % 6
    f = hi - hif
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = np.zeros((col,row)) , np.zeros((col,row)) , np.zeros((col,row))
    for i in range(0,col):
        for j in range(0,row):
            if hif[i,j] == 0:
                r[i,j] , g[i,j] , b[i,j] = v[i,j] , t[i,j] , p[i,j]
            elif hif[i,j] == 1:
                r[i,j] , g[i,j] , b[i,j] = q[i,j] , v[i,j] , p[i,j]
            elif hif[i,j] == 2:
                r[i,j] , g[i,j] , b[i,j] = p[i,j] , v[i,j] , t[i,j]
            elif hif[i,j] == 3:
                r[i,j] , g[i,j] , b[i,j] = p[i,j] , q[i,j] , v[i,j]
            elif hif[i,j] == 4:
                r[i,j] , g[i,j] , b[i,j] = t[i,j] , p[i,j] , v[i,j]
            elif hif[i,j] == 5:
                r[i,j] , g[i,j] , b[i,j] = v[i,j] , p[i,j] , q[i,j]
    return r, g, b
def rgb2utf8(rgb):
    """
    转换数据格式为utf-8
    输入：numpyArray格式三维数组
    返回：utf-8格式可出图三维数组
    """
    min_val = np.min(rgb.ravel())
    max_val = np.max(rgb.ravel())
    RGB = np.uint8((rgb.astype(np.float) - min_val) / (max_val - min_val) * 255)
    return RGB
def HSV(data_low,data_high):
    """
    使用opencv将色彩空间RGB转为HSV,H色调,S饱和度,V明度
    输入：numpyArray格式的三维数组
    返回：可绘出图像的utf-8格式的三维数组
    """
    min_val = np.min(data_low.ravel())
    max_val = np.max(data_low.ravel())
    data_low = np.uint8((data_low.astype(np.float) - min_val) / (max_val - min_val) * 255)
    H_low , S_low , V_low = rgb2hsv(data_low[0],data_low[1],data_low[2])
    H_high  ,S_high , V_high = rgb2hsv(data_high[0],data_high[1],data_high[2])
    R,G,B = hsv2rgb(H_low,S_low,V_high)
    R = rgb2utf8(R)
    G = rgb2utf8(G)
    B = rgb2utf8(B)
    merged = cv2.merge([R,G,B])
    RGB = Image.fromarray(merged)
    return RGB
def main(path_low,path_high):
    data_low = gdal_open(path_low)
    data_high = gdal_open(path_high)
    data_low = imresize(data_low,data_high)
    RGB = HSV(data_low,data_high)
    RGB.save(r'C:\Users\64908\Desktop\HSV.png','png')
if __name__ == "__main__":
    path_low = r'C:\Users\64908\Desktop\RGB.tif'
    path_high = r'C:\Users\64908\Desktop\Band8.tif'
    main(path_low,path_high)