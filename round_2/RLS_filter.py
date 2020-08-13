# coding=utf-8
'''
Created on 2020-6-20
@author: jiangao
Project: 递推最小二乘自适应数字滤波
'''
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import Signal_simulation
import copy
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
def RLS(x,y,M):
    """
    递推最小二乘法自适应数字滤波
    x为一维数组
    """
    n = x.shape[0]
    m = M - 1
    p = np.zeros(m*m)
    px = np.zeros(m)
    u = np.zeros(m)
    g = np.zeros(m)
    w = np.zeros(m)
    d = y

    # d[0] = 2.0*x[0]
    # d[1] = 2.0*x[0] - 0.5*x[1]
    # d[2] = 2.0*x[0] - 0.5*x[1] + 1.4*x[2]
    # for i in range(3,n):
    #     d[i] = 2.0*x[i] - 0.5*x[i-1] + 1.4*x[i-2] + 0.1*x[i-3]

    r = 1.0
    for i in range(m):
        for j in range(m):
                p[i*m+j] = 0.0
    for i in range(m):
        p[i*m+i] = 1.0e+8
    for k in range(n):
        px[0] = x[k]
        for j in range(m):
            u[j] = 0.0
            for i in range(m):
                u[j] = u[j] + (1/r)*p[j*m+i]*px[i]
        s = 1.0
        for i in range(m):
            s = s + u[i] * px[i]
        for i in range(m):
            g[i] = u[i]/s
        x[k] = 0.0
        for i in range(m):
            x[k] = x[k] + w[i] * px[i]
        a = d[k] - x[k]
        for i in range(m):
            w[i] = w[i] + g[i] * a
        for j in range(m):
            for i in range(m):
                p[j*m+i] = (1/r) *p[j*m+i] - g[j] * u[i]
        for i in range(m-1,0,-1):
            px[i] = px[i-1]
    return d , x
if __name__ == "__main__":
    a = Signal_simulation.electromagnetic_signal(1,1,500)
    x = a.electromagnetic()
    v = Signal_simulation.Gaussian_noise(0,0.5,250)
    b = v.gaussian()
    g = Signal_simulation.Sum(b,x)
    c = copy.deepcopy(g)
    d , y = RLS(g,x,5)

    # plt.plot(d,label = '理想输出信号')
    plt.plot(c,label = '实际输入信号')
    plt.plot(y,label = '实际输出信号')
    plt.legend()
    plt.grid()
    plt.show()