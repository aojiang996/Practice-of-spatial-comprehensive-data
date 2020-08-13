#!/usr/bin/env python
# encoding: utf-8
'''
@author: zxqyiyang
@contact: 632695399@qq.com
@file: noise_and_signal.py
@time: 2020/6/21 22:13
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Guassian_noise():
    def __init__(self,mean, scale, count):
        self.mean = mean
        self.scale = scale
        self.count = count
    def generate(self):
        data = np.random.normal(self.mean, self.scale, self.count)
        return data

class random_noise():
    def __init__(self,count):
        self.count = count
    def generate(self):
        data = np.sin(np.random.rand(self.count)*math.pi*2)/10.0
        return data

class Sky_electric_nosie():
    def __init__(self,count):
        self.count = count
    def generate(self, guassian_noise):
        p, q = 2, 2
        a = [1.0, 1.45, 0.6]
        b = [1.0, -0.2,-0.1]
        """
        x[n] + a1*x[n-1] + a2*x[n-2] = b1*w[n-1]+b2*w[n-2]
        """
        x = np.zeros(self.count)
        x[0] = b[0] * guassian_noise[0]
        for n in range(1, self.count):
            s = 0
            for i in range(1, p+1):
                if(n==1):
                    s += -a[1]*x[n-1] + b[1]*guassian_noise[n-1]
                    break
                else:
                    s += -a[i] * x[n - i] + b[i] * guassian_noise[n - i]
            x[n] = s
        return x

class Signal():
    def __init__(self, count):
        self.count = count
    def generate(self):
        N = [i for i in range(1, 100)]
        data = np.zeros(self.count)
        for i in range(1, self.count+1):
            x = 0
            for n in N:
                x += math.exp(-(n ** 2) * i / 10)
            data[i-1] = x*4
        return data

class SNP():
    def calculation_power(list):
        """:param list: 信号列表 """
        x = np.array(list)
        power = sum(x * x) / np.size(x)
        return power
    def calculation_SNP(PS, PN):
        SNR = 10 * math.log10(PS / PN)
        return SNR

class Calculation_fft():
    def __init__(self,data):
        self.data = data
    def fft(self):
        """Compute the discrete Fourier Transform of the 1D array x"""
        signal = np.asarray(self.data, dtype=float)
        N = self.data.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.abs(np.dot(M, self.data))

class Filter():
    def __init__(self,data):
        self.data = data
    def Kalman_calculation(self):
        shape = self.data.shape[0]
        x = np.zeros(shape)
        x_= np.zeros(shape)
        p_= np.zeros(shape)
        k = np.zeros(shape)
        p = np.zeros(shape)
        r, q = 0.000000001, 0.000000001
        p[0], x[0] = 1.0, self.data[0]
        for i in range(1, shape):
            x_[i]= x[i - 1]
            p_[i]= p[i - 1] + q
            k[i] = p_[i]/(p_[i] + r)
            x[i] = x_[i] + k[i] * (self.data[i] - x_[i])
            p[i] = (1 - k[i]) * p_[i]
        return x

    def fda2(self): #（输入的信号，限制频率）
        b, a = signal.butter(8,2.0*100/1000, 'lowpass')
        data = signal.filtfilt(b, a,self.data)
        return data

    def median(self):
        width = 3
        a = np.zeros(self.data.shape[0] + math.ceil(width/2))
        y = np.zeros(self.data.shape[0])
        a[0], a[-1], a[1:-1] = self.data[0], self.data[-1], self.data
        for i in range(self.data.shape[0]):
            y[i] = np.median(a[i:i+width])
        return y

    def mean(self):
        width = 3
        a = np.zeros(self.data.shape[0] + math.ceil(width / 2))
        y = np.zeros(self.data.shape[0])
        a[0], a[-1], a[1:-1] = self.data[0], self.data[-1], self.data
        for i in range(self.data.shape[0]):
            y[i] = np.mean(a[i:i+width])
        return y

def signal_noise_sum(data1, data2):
    if(data1.shape == data2.shape):
        data = np.array(data1) + np.array(data2)
        return data
    else:
        print("data1 shape is different from data2")

def main():
    mean, sclae, count = 0, 1, 200
    time = [i/100 for i in range(count)]

    guassian_noise = Guassian_noise(mean, sclae, count).generate()
    print(guassian_noise.shape)

    sky_electric_nosie = Sky_electric_nosie(count).generate(guassian_noise)
    print(sky_electric_nosie.shape)

    signals = Signal(count).generate()
    print(signals.shape)

    data = signal_noise_sum(signals, sky_electric_nosie)

    fft = Calculation_fft(data).fft()

    k_filter_data = Filter(data).Kalman_calculation()

    fir_filter_data = Filter(data).fda2()

    median_filter_data = Filter(data).median()

    mean_filter_data = Filter(data).mean()
    # plt.plot(time, sky_electric_nosie)
    # plt.plot(time, data, linestyle="-", label="hunhe", color="red")
    # plt.plot(time, signal, color="black")
    # plt.plot(time, fft)
    # plt.plot(time, k_filter_data, linestyle="-", label="lvbo", color="blue")
    # plt.plot(time, fir_filter_data, color="red")
    # plt.plot(time, median_filter_data)
    plt.plot(time, mean_filter_data)
    plt.show()


if __name__ == '__main__':
    main()

