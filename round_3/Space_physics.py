# coding=utf-8
'''
Created on 2020-6-23
@author: jiangao
Project: 空间物理实验
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import spacepy.time as spt
import spacepy.coordinates as spc
from spacepy import pycdf
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

ax = plt.figure().add_subplot(111,projection='3d')#获取绘制三D图像的类
os.environ["CDF_LIB"] = "D:\CDF\lib"#打开CDF阅读环境

def read_MFI(path):
    """
    读取CDF数据，获取时间，磁场强度分量，与磁场总强度
    输入：CDF文件路径
    输出：字典类型数据
    """
    data = pycdf.CDF(path)
    MFI = {
        "EPOCH":data["Epoch"][:],
        "BX":data["BX"][:],
        "BY":data["BY"][:],
        "BZ":data["BZ"][:],
        "BTOT":data["BT"][:]
    }
    return MFI

def read_FPE(path):
    """
    读取CDF数据，获取时间，等离子体密度，能量密度，温度，GSE坐标分量，速度
    输入：CDF文件路径
    输出：字典类型数据
    """
    data = pycdf.CDF(path)
    FPE = {
        "EPOCH":data["Epoch"][:],
        "PROTON_DENSITY":data["DEN"][:],
        "ENERGY_DENSITY":data["ENDEN"][:],
        "TEMP":data["T"][:],
        "X_GSE":data["GSEX"][:],
        "Y_GSE":data["GSEY"][:],
        "Z_GSE":data["GSEZ"][:],
        "VX_BULK_2D":data["VX"][:],
        "VY_BULK_2D":data["VY"][:]
    }
    return FPE

def smooth(array,size):
    """
    数据平滑处理
    输入：np.ndarray格式数组
    输出：平滑后的np.ndarray格式数组
    """
    array_ = np.ones(int(size)) / float(size)
    out = np.convolve(array, array_, 'same')# numpy的卷积函数
    for i in range(size):
        out[i] = array[i]
    return out  

def GSE2GSM(FPE):
    """
    GSE转GSM函数
    输入：字典格式数据
    输出：np.ndarray格式数据
    """
    GSM = np.zeros((3,FPE["X_GSE"].shape[0]))
    for i in range(GSM.shape[1]):
        SM = spc.Coords(
            [[
                FPE["X_GSE"][i],
                FPE["Y_GSE"][i],
                FPE["Z_GSE"][i]
            ]],
            'GSE','car'
        )
        SM.ticks = spt.Ticktock(FPE["EPOCH"][i],'ISO')
        SM = SM.convert('GSM','car')
        GSM[0][i] = SM.data[0][0]
        GSM[1][i] = SM.data[0][1]
        GSM[2][i] = SM.data[0][2]
    return GSM

def GSE2GSM_B(MFI):
    """
    GSE转GSM函数
    输入：字典格式数据
    输出：np.ndarray格式数据
    """
    GSM = np.zeros((3,len(MFI["BZ"])))
    for i in range(GSM.shape[1]):
        SM = spc.Coords(
            [[
                MFI["BX"][i],
                MFI["BY"][i],
                MFI["BZ"][i]
            ]],
            'GSE','car'
        )
        SM.ticks = spt.Ticktock(MFI["EPOCH"][i],'ISO')
        SM = SM.convert('GSM','car')
        GSM[0][i] = SM.data[0][0]
        GSM[1][i] = SM.data[0][1]
        GSM[2][i] = SM.data[0][2]
    return GSM

def _time_(MFI):
    """
    提取相同时间
    输入：字典格式数据
    输出：字典格式数据
    """
    _MFI_ = {
        "EPOCH":[],
        "TIME":[],
        "BX":[],
        "BY":[],
        "BZ":[],
        "BTOT":[]
    }
    
    for i in range(len(MFI["EPOCH"])):
        if MFI["EPOCH"][i].hour == MFI["EPOCH"][i-1].hour and MFI["EPOCH"][i].minute == MFI["EPOCH"][i-1].minute:
            continue
        BX = 0.0
        BY = 0.0
        BZ = 0.0
        BTOT = 0.0
        k = 0
        for j in range(len(MFI["EPOCH"])):
            if MFI["EPOCH"][j].hour == MFI["EPOCH"][i].hour and MFI["EPOCH"][j].minute == MFI["EPOCH"][i].minute:
                BX += MFI["BX"][j]
                BY += MFI["BY"][j]
                BZ += MFI["BZ"][j]
                BTOT += MFI["BTOT"][j]
                k += 1
        BX = BX / k
        BY = BY / k
        BZ = BZ / k
        BTOT = BTOT / k
        _MFI_["EPOCH"].append(MFI["EPOCH"][i])
        _MFI_["TIME"].append(MFI["EPOCH"][i].hour + MFI["EPOCH"][i].minute*0.01*100/60)
        _MFI_["BX"].append(BX)
        _MFI_["BY"].append(BY)
        _MFI_["BZ"].append(BZ)
        _MFI_["BTOT"].append(BTOT)
    return _MFI_

def filt(data):
    x = []
    y = []
    for i in range(len(data)):
        if data[i] <= 150 and data[i] >= -150:
            y.append(data[i])
            x.append(i)
    return x , y
def fori(data,x):
    array = []
    for i in x:
        array.append(data[i])
    return array
def filt2(data):
    x = []
    y = []
    for i in range(len(data)):
        if data[i] <= 1000 and data[i] >= -1000:
            y.append(data[i])
            x.append(i)
    return x , y
def filt3(data):
    x = []
    y = []
    for i in range(len(data)):
        y.append(data[i])
        x.append(i)
    return x , y

if __name__ == "__main__":
    MFI_path = r"D:\experiment\Practice of spatial comprehensive data\round_3\isee2_4sec_mfi_19780812_v01.cdf"
    FPE_path = r"D:\experiment\Practice of spatial comprehensive data\round_3\isee2_h1_fpe_19780812_v01.cdf"
    MFI = read_MFI(MFI_path)
    FPE = read_FPE(FPE_path)

    _MFI_ = _time_(MFI)
    GSM = GSE2GSM(FPE)
    area = np.pi * 2**2
    B_GSM = GSE2GSM_B(_MFI_)
    # time_fpe = time(FPE)

    # ax.scatter(GSM[0],GSM[1],GSM[2])
    # plt.legend()
    # ax.set_xlabel('gsmx')
    # ax.set_ylabel('gsmy')
    # ax.set_zlabel('gsmz')
    # plt.show()



    # plt.subplot(4,1,1)
    # plt.grid()
    # x , y  = filt2(FPE["VX_BULK_2D"])
    # time = fori(FPE["EPOCH"],x)
    # plt.plot(time,smooth(y,2),label = u'VX')
    # plt.ylabel(u'等离子体流速VX\n(km/s)')
    # plt.legend()

    plt.subplot(3,1,1)
    plt.grid()
    x , y  = filt3(FPE["PROTON_DENSITY"])
    time = fori(FPE["EPOCH"],x)
    plt.plot(time,smooth(y,2),label = u'PROTON_DENSITY')
    plt.ylabel(u'等离子体密度\n(cm-3)')
    plt.legend()

    # plt.subplot(4,1,2)
    # plt.grid()
    # x , y  = filt3(FPE["TEMP"])
    # time = fori(FPE["EPOCH"],x)
    # plt.plot(time,smooth(y,2),label = u'TEMP')
    # plt.ylabel(u'温度\n(Deg_Kelvin)')
    # plt.legend()
    
    # plt.subplot(4,1,3)
    # plt.grid()
    # x , y  = filt3(FPE["PROTON_DENSITY"])
    # time = fori(FPE["EPOCH"],x)
    # plt.plot(time,smooth(y,2),label = u'等离子体密度')
    # plt.ylabel(u'等离子体密度\n(cm-3)')
    # plt.legend()

    plt.subplot(3,1,2)
    plt.grid()
    x , y  = filt3(FPE["TEMP"])
    time = fori(FPE["EPOCH"],x)
    plt.plot(time,smooth(y,2),label = u'TEMP')
    plt.ylabel(u'TEMP\n(Deg_Kelvin)')
    plt.legend()

    plt.subplot(3,1,3)
    plt.grid()
    x , y  = filt(_MFI_["BTOT"])
    time = fori(_MFI_["EPOCH"],x)
    plt.plot(time,smooth(y,2),label = u'BTOT')
    plt.ylabel(u'BTOT\n(nT)')
    plt.legend()

    plt.show()