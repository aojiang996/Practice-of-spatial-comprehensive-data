from osgeo import gdal, gdalconst
import numpy as np
import cv2
import os
import scipy.misc as smi
from PIL import Image  

def im2image(im):#转换格式
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype(np.float) - min_val) / (max_val - min_val)#将像素值换成0~1的值
    out = out*255   #乘以255，像素值换成颜色值
    out = np.uint8(out)#utf-8编码格式转换
    return out

def brovey(p1,p2):
    # 基于Brovey变换融合算法：p1代表彩色的多光谱图像；p2代表全色波段图像
    dataset = gdal.Open(p1)
    dataup = gdal.Open(p2)
    
    col = dataset.RasterXSize#低分图像长度
    row = dataset.RasterYSize#低分图像宽度
    cols = dataup.RasterXSize#高分图像长度
    rows = dataup.RasterYSize#高分图像宽度

    dpup = dataup.ReadAsArray(0,0,cols,rows)
    dpup = dpup.astype(np.float)

    band = dataset.GetRasterBand(1)#取第三波段
    r=band.ReadAsArray(0,0,col,row)#从数据的起始位置开始，取col行row列数据
    dbr = r.astype(np.float)

    band = dataset.GetRasterBand(2)
    g=band.ReadAsArray(0,0,col,row)
    dbg = g.astype(np.float)

    band = dataset.GetRasterBand(3)
    b=band.ReadAsArray(0,0,col,row)
    dbb = b.astype(np.float)

    w_, h_ =  cols , rows
    im_r = smi.imresize(dbr, (h_,w_),interp='bicubic')#重采样
    im_g = smi.imresize(dbg, (h_,w_),interp='bicubic')
    im_b = smi.imresize(dbb, (h_,w_),interp='bicubic')
    im_r = im_r.astype(np.float)
    im_g = im_g.astype(np.float)
    im_b = im_b.astype(np.float)

    im_ = np.array((r,g,b))
    im_dp = np.array((im_r,im_g,im_b))
    im_dp = im_dp.astype(np.float)
    print(im_dp.shape)
    print(im_dp)
    x = np.zeros((h_,w_))
    p = np.zeros(((3,h_,w_)))
    x = im_dp[0] + im_dp[1] + im_dp[2]
    print(x)
    p[0]= im_dp[0]*dpup/x
    p[1]=np.multiply(np.true_divide(
        im_dp[1],x),dpup)
    p[2]=np.multiply(np.true_divide(
        im_dp[2],x),dpup)
    
    # for i in range(h_):#最近邻法融合
    #     for j in range(w_):
    #         x[i][j] = im_dp[0][i][j] + im_dp[1][i][j] + im_dp[2][i][j]
    #         p[0][i][j] = im_dp[0][i][j] * dpup[i][j] / x[i][j]
    #         p[1][i][j] = im_dp[1][i][j] * dpup[i][j] / x[i][j]
    #         p[2][i][j] = im_dp[2][i][j] * dpup[i][j] / x[i][j]

    
    p = im2image(p)

    merged = cv2.merge([p[0],p[1],p[2]])#融合三色道

    new_map = Image.fromarray(merged)
    new_map.save(r'D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\1.png','png')

    
    
    

    

if __name__ == "__main__":
    p1 = r"D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\mif.tif"
    p2 = r"D:\01近期文档\空科综合实习内容一（gis-rs）候选题目\01次实验\B8.tif"   
    
    brovey(p1,p2)
   
    
    