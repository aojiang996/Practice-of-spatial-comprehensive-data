from osgeo import gdal, gdalconst
import numpy as np
import cv2
import os
import scipy.misc as smi
from PIL import Image,ImageEnhance

def im2image(im):#转换格式
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype(np,float) - min_val) / (max_val - min_val)#将像素值换成0~1的值
    out = out*255   #乘以255，像素值换成颜色值
    out = np.uint8(out)#utf-8编码格式转换
    return out

def IHS(p1,p2):
# 基于IHS变换融合算法：p1代表彩色的多光谱图像；p2代表全色波段图像
    dataset = gdal.Open(p1)
    dataup = gdal.Open(p2)
    
    col = dataset.RasterXSize#低分图像长度
    row = dataset.RasterYSize#低分图像宽度
    cols = dataup.RasterXSize#高分图像长度
    rows = dataup.RasterYSize#高分图像宽度

    dpup = dataup.ReadAsArray(0,0,cols,rows)
    dpup = dpup.astype(np.float)

    band = dataset.GetRasterBand(3)#取第三波段
    r=band.ReadAsArray(0,0,col,row)#从数据的起始位置开始，取200行200列数据
    dbr = r.astype(np.float)

    band = dataset.GetRasterBand(2)
    g=band.ReadAsArray(0,0,col,row)
    dbg = g.astype(np.float)

    band = dataset.GetRasterBand(1)
    b=band.ReadAsArray(0,0,col,row)
    dbb = b.astype(np.float)

    A = np.zeros((3,3))
    B = np.zeros((3,3))
    A = A.astype(np.float)
    B = B.astype(np.float)
    A = np.array(A)
    B = np.matrix(B)

    A = [[1./np.sqrt(3),1./np.sqrt(3),1./np.sqrt(3)],[1./np.sqrt(6),1./np.sqrt(6),-2./np.sqrt(6)],[1./np.sqrt(2),-1./np.sqrt(2),0.]] 
    #RGB－>IHS正变换矩阵
    B = [[1./np.sqrt(3),1./np.sqrt(6),1./np.sqrt(2)],[1./np.sqrt(3),1./np.sqrt(6),-1./np.sqrt(2)],[1./np.sqrt(3),-2./np.sqrt(6),0.]] 
    #IHS－>RGB逆变换矩阵

    A = np.matrix(A)
    B = np.matrix(B)

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

    v1 = np.zeros((3,1))
    v2 = np.zeros((3,1))
    RGB = np.zeros((3,cols,rows))
    RGB = RGB.astype(np.float)

    for i in range(cols):
        for j in range(rows):
            v1[0] = dpup[i][j]
            v1[1] = dpup[i][j]
            v1[2] = dpup[i][j]
            v2[0] = im_dp[0][i][j]
            v2[1] = im_dp[1][i][j]
            v2[2] = im_dp[2][i][j]

            u1 = A*v1
            u2 = A*v2
            u2[1] = u1[1]
            v2 = B*u2

            RGB[0][i][j]=v2[0]    #逆变换
            RGB[1][i][j]=v2[1]
            RGB[2][i][j]=v2[2]
    
    p = im2image(RGB)#转换格式
    R = p[0]
    G = p[1]
    B = p[2]



    merged = cv2.merge([p[0],p[1],p[2]])#融合三色道
    new_map = Image.fromarray(merged)
    new_map.show()
    # enhance = image_enhance(new_map, 3, 1.5) #调用增强类类

    # img_contrasted = enhance.image_contrasted()
    # img_contrasted.show()
    # definition(R,G,B,cols,rows)

class image_enhance():
    """
    图像增强类：包括亮度和对比度
    """
    def __init__(self, img, brightness, contrast):
        self.img = img
        self.brightness = brightness
        self.contrast = contrast
    def image_brightened(self):
        enh_bri = ImageEnhance.Brightness(self.img)
        image_brightened = enh_bri.enhance(self.brightness)
        return image_brightened

    def image_contrasted(self):
        enh_con = ImageEnhance.Contrast(self.img)
        img_contrasted = enh_con.enhance(self.contrast)
        return img_contrasted

def definition(r,g,b,cols,rows):
    r_ = 0
    g_ = 0
    b_ = 0
    for i in range(cols-1):
        for j in range(rows-1):
            r_=r_ + np.sqrt(((r[i + 1][j] - r[i][j])^2 + (r[i][j + 1] - r[i][j])^2)/2)
            g_=g_ + np.sqrt(((g[i + 1][j] - g[i][j])^2 + (g[i][j + 1] - g[i][j])^2)/2)
            b_=b_ + np.sqrt(((b[i + 1][j] - b[i][j])^2 + (b[i][j + 1] - b[i][j])^2)/2)
    _r = r_/(cols-1)/(rows-1)
    _g = g_/(cols-1)/(rows-1)
    _b = b_/(cols-1)/(rows-1)
    print("R的清晰度为：{0:.4}，G的清晰度为：{1:.4}，B的清晰度为：{2:.4}".format(_r,_g,_b))


if __name__ == "__main__":
    p1 = "D:/01近期文档/空科综合实习内容一（gis-rs）候选题目/01次实验/RGB.tif"
    p2 = "D:/01近期文档/空科综合实习内容一（gis-rs）候选题目/01次实验/Band8.tif"   
    
    IHS(p1,p2)
    
    
