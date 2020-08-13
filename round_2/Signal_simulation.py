# coding=utf-8
'''
Created on 2020-6-18
@author: jiangao
Project: 卡尔曼滤波
'''
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

class signal():
    """
    产生航空瞬变电磁信号的类
    """
    def __init__(self,τ,c,pad):
        self.τ = τ
        self.c = c
        self.pad = pad
    def Astroelectric(self):
        sums = 0
        t = np.linspace(self.c - 6 * self.τ , self.c + 6 * self.τ , self.pad)
        n = 1000
        for i in range(1,n+1):
            sums += np.exp(-(i*i)*(t / self.τ))
        curve = (self.c / self.τ) * sums
        curve = np.nan_to_num(curve)
        astroelectric = np.zeros(self.pad)
        j = 0
        for i in range(len(curve)):# 筛选电磁信号无限大值
            if curve[i] != 0 and curve[i] < 1000:
                astroelectric[j] = curve[i]
                j += 1
            if j == self.pad:
                break
        return astroelectric
class noise():
    """
    产生噪声类
    """
    def __init__(self,μ,sigma,pad):
        self.μ = μ
        self.sigma = sigma
        self.pad = pad
    def Gaussian(self):
        """
        高斯噪声
        """
        gaussian = np.random.normal(self.μ, self.sigma, self.pad) 
        return gaussian
    def ARMA(self):
        """
        天电噪声
        """
        gaussian = self.Gaussian()
        p , q = 2 , 2 
        #ARMA(2,2)中自回归系数ai = {1.0 , 1.45 , 0.6} 滑动平均系数bi = {1.0 , -0.2 , -0.1}
        a = [1.0 , 1.45 , 0.6]
        b = [1.0 , -0.2 , -0.1]
        x = np.zeros(self.pad)
        x[0] = b[0] * gaussian[0]
        #ARMA(2,2)模型为 x(k) + 1.45x(k-1) +0.6x(k-2) = gaussian(k) -0.2w(k) - 0.1w(k) gaussian(k)为高斯噪声, x(k)为天电噪声
        for n in range(1, self.pad):
            s = 0.0
            for i in range(1, p+1):
                if(n==1):
                    s += b[1]*gaussian[n-1] - a[1]*x[n-1]
                    break
                else:
                    s += b[i] * gaussian[n - i] - a[i] * x[n - i]
            x[n] = s
        return x
    def Random(self):
        random = np.random.uniform(size = self.pad)
        return  random
class Spectrum_analysis():
    """
    频谱分析类
    """
    def __init__(self,wave):
        self.wave = wave
        self.signal = signal
    def analysis(self):
        """
        频谱分析
        """
        wave = np.nan_to_num(self.wave)
        wave_ = []
        j = 0                 
        for i in wave:                         # 筛选信号无限大值
            if i != 0 and i < 1000:
                wave_.append(i)
                j += 1
            if j == self.wave.size:
                break
        sampling_rate = 2000
        fft_size = 128                         # FFT处理的取样长度
        t = x = np.linspace(0,1,128)
        x = wave_
        xs = x[:fft_size]                      # 从波形数据中取样fft_size个点进行运算
        xf = np.fft.fft(xs)/fft_size           # 利用np.fft.fft()进行FFT计算，fft()是为了更方便对实数信号进行变换
        freqs = np.linspace(0, sampling_rate, fft_size)
        # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。
        # 为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
        xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        # # 绘图显示结果
        plt.figure(figsize=(8,4))
        plt.subplot(211)
        plt.plot(t[:fft_size], xs)
        plt.grid()
        plt.xlabel(u"Time(S)")
        plt.title(u"WaveForm And Freq")
        plt.subplot(212)
        plt.plot(freqs, xfp)
        plt.xlabel(u"Freq(Hz)")
        plt.subplots_adjust(hspace=0.4)
        plt.grid()
        plt.show()
    def snp(self,ps,pn):
        return 10 * math.log10((sum(abs(ps)**2) / ps.size) / (sum(abs(pn)**2) / pn.size))
    def SNP(self,mix_gaussian,mix_random,mix_arma,gaussian_,random_,arma_,gaussian,random,arma):
        SNP_gaussian = self.snp(mix_gaussian , gaussian)
        SNP_random = self.snp(mix_random , random)
        SNP_arma = self.snp(mix_arma , arma)
        SNP_gaussian_wave = self.snp(gaussian_ , gaussian)
        SNP_random_wave = self.snp(random_ , random)
        SNP_arma_wave = self.snp(arma_ , arma)
        print('{:{}<8}{:{}>8}{:>7.4f}{:{}>8}{:>7.4f}{:{}>8}{:>7.4f}'.format('滤波前的信噪比：',chr(12288),'高斯噪声：',chr(12288),SNP_gaussian,'随机噪声：',chr(12288),SNP_random,'天电噪声：',chr(12288),SNP_arma))
        print('{:{}<8}{:{}>8}{:>7.4f}{:{}>8}{:>7.4f}{:{}>8}{:>7.4f}'.format('滤波后的信噪比：',chr(12288),'高斯噪声：',chr(12288),SNP_gaussian_wave,'随机噪声：',chr(12288),SNP_random_wave,'天电噪声：',chr(12288),SNP_arma_wave))

class wave_filter():
    """
    滤波类
    """
    def __init__(self,signal):
        self.signal = signal
    def Sum(self,noise):
        """
        混合函数
        """
        sums = np.zeros(noise.size)
        for i in range(len(noise)):
            sums[i] = noise[i] + self.signal[i]
        return sums

    def kalman(self,wave,noise):
        """
        卡尔曼滤波器
        """
        n = wave.size
        Q , R = 1e-10, 1
        x = np.zeros(n)#滤波前值
        p = np.zeros(n)#误差的协方差
        x_pro = np.zeros(n)#滤波后值
        p_pro = np.zeros(n)#误差的协方差
        K = np.zeros(n)#卡尔曼增益
        x[0] = wave[0]#给定初值，然后迭代
        p[0] = 1.0

        for i in range(1,n):
            x_pro[i] = x[i-1]
            p_pro[i] = p[i-1] + Q 
            K[i] = p_pro[i] / (p_pro[i] + R) #卡尔曼增益
            x[i] = x_pro[i] + K[i] * (wave[i] - x_pro[i])
            p[i] = (1 - K[i]) * p_pro[i]#更新迭代器

        snp1 = Spectrum_analysis(x).snp(self.signal,noise)
        snp2 = Spectrum_analysis(x).snp(wave,noise)

        plt.title(u'卡尔曼滤波器')
        plt.plot(wave,label = '观测值',linestyle = '--')     
        plt.plot(x,label = '滤波后') 
        plt.plot(self.signal,label = '预测值') 
        plt.text(10 , x[0] , u'滤波前信噪比{:8.4f}'.format(snp1))
        plt.text(10 , x[0]-0.5 , u'滤波后信噪比{:8.4f}'.format(snp2))
        plt.text(10,x[0]-1,u'Q = {},R = {}'.format(Q,R))
        plt.xlabel(u"time\n(s)")
        plt.grid()
        plt.legend()
        plt.show()
        return x

    def MeanFilter(self,wave,pad = 1):
        """
        均值滤波
        """
        waves = []
        a = np.zeros(pad*2+1)
        for i in range(pad):
            waves.append(wave[i])
        for i in range(pad,len(wave)-pad):
            a[:pad] = waves[i-pad:]
            a[pad:] = wave[i:i+pad+1]
            b = 0
            for j in range(len(a)):
                b += a[j]
            waves.append(b/len(a))
        for i in range(pad):
            waves.append(wave[len(wave)-pad+i])
        return waves

    def MedianFilter(self,wave,pad = 1):
        """
        中值滤波
        """
        waves = []
        a = np.zeros(pad*2+1)
        for i in range(pad):
            waves.append(wave[i])
        for i in range(pad,len(wave)-pad):
            a[:pad] = waves[i-pad:]
            a[pad:] = wave[i:i+pad+1]
            a.sort()
            waves.append(a[pad])
        for i in range(pad):
            waves.append(wave[len(wave)-pad+i])
        return waves

    def FourierFilter(self,wave):
        """
        傅里叶变换
        """
        x = np.linspace(0,1,wave.size)
        fft1 = abs(np.fft.fft(wave)) / ((len(x) / 2))# 归一化处理
        # 由于对称性，只取一半区间
        fft = np.array(self.MeanFilter(np.kron(np.array(fft1[range(int(len(x)/2))]) , np.ones(2))))*4
        return fft
    def draw(self,wave):
        median = self.MedianFilter(wave)
        fft = self.FourierFilter(wave)
        mean = self.MeanFilter(wave)

        plt.title(u'滤波')
        plt.plot(wave,label = u'滤波前',linestyle = '--')
        plt.plot(mean,label = u'均值滤波后')
        plt.plot(median,label = u'中值滤波后')
        plt.plot(fft,label = u'傅里叶变换后')
        plt.plot(astroelectric,label = u'原始电磁信号')
        plt.xlabel(u'time\n(s)')
        plt.grid()
        plt.legend()
        plt.show()

def Sum(signal,noise):
        """
        混合函数
        """
        sums = np.zeros(noise.size)
        for i in range(len(noise)):
            sums[i] = noise[i] + signal[i]
        return sums

if __name__ == "__main__":
    noise = noise(0,0.5,250)#噪声
    signal = signal(0.5,0.5,250)
    astroelectric = signal.Astroelectric()#产生电磁信号
    analysis = Spectrum_analysis(astroelectric)
    mix = wave_filter(astroelectric)

    gaussian = noise.Gaussian()#高斯噪声
    random = noise.Random()#随机噪声
    arma = noise.ARMA()#天电噪声

    Spectrum_analysis(gaussian).analysis()#频谱分析
    Spectrum_analysis(random).analysis()
    Spectrum_analysis(arma).analysis()
    
    mix_gaussian = mix.Sum(gaussian)#混合噪声
    mix_random = mix.Sum(random)
    mix_arma = mix.Sum(arma)

    mix_all = Sum(mix_gaussian,random)
    mix_all = Sum(mix_all,arma)
    Spectrum_analysis(mix_all).analysis()
    noise_all = Sum(random,arma)
    noise_all = Sum(noise_all,gaussian)

    alls = mix.kalman(mix_all,noise_all)
    Spectrum_analysis(alls).analysis()
    gaussian_ = mix.kalman(mix_gaussian,gaussian)#卡尔曼滤波
    random_ = mix.kalman(mix_random,random)
    arma_ = mix.kalman(mix_arma,arma)

    Spectrum_analysis(gaussian_).analysis()#频谱分析
    Spectrum_analysis(random_).analysis()
    Spectrum_analysis(arma_).analysis()
    analysis.SNP(mix_gaussian,mix_random,mix_arma,gaussian_,random_,arma_,gaussian,random,arma)#统计信噪比

    mix.draw(gaussian_)#中值滤波、均值滤波、傅里叶变换
    mix.draw(random_)
    mix.draw(arma_)