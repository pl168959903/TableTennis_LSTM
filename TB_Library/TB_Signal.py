import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

'''
多項式去趨勢
'''
def Detrends(dataArray, n):
    lenArray = np.arange(len(dataArray)).reshape(-1, 1)
    pf = preprocessing.PolynomialFeatures(degree=n)
    Xp = pf.fit_transform(lenArray)

    model = linear_model.LinearRegression()
    model.fit(Xp, dataArray)

    retval = dataArray - model.predict(Xp)
    return retval

'''
正規化處理等比例縮放至[0,1]
'''
def MaxMin(arrayData):
    model = preprocessing.MinMaxScaler()
    retval = model.fit_transform(arrayData)
    return retval

'''
數位低通濾波器
'''
def LowpassFilter(data, cutoff, fs=85, order=1):
    inputData = data.squeeze()
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    y = filtfilt(b, a, inputData)
    return y.reshape(-1, 1)

'''
取得FFT處理後最大的數值
'''
def GetFFTMaxValue(signal, fq=85, showFFT=False):
    inputSignal = signal.squeeze()
    sampleSize = len(inputSignal)

    ffty = fft(inputSignal)
    fftx = fftfreq(sampleSize, 1/fq)[:sampleSize//2]
    fftg = 2.0/sampleSize * np.abs(ffty[0:sampleSize//2])

    fftLocX = fftx[np.argmax(fftg)]
    fftLocY = np.max(fftg)
    if showFFT == True:
        plt.figure(figsize=(15, 3))
        plt.grid()
        plt.plot(fftx, fftg, label="FFT")
        plt.annotate(
            'FFT Max Value. freq: %.3f' % (fftLocX), xy=(fftLocX, fftLocY),
            xytext=(fftLocX+3, fftLocY*0.8),
            xycoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05)
        )
        plt.show()

    return fftLocX

'''
建立二值化訊號
'''
def CreatBinSigal(signal, valve):

    retval = np.zeros((len(signal), 1))
    arrLen = np.arange(0, len(signal), 1)

    for i in arrLen:
        if signal[i] > valve:
            retval[i] = 1
        else:
            retval[i] = 0
    return retval

'''
尋找二值化訊號中心點
'''
def FindBinSigalCerter(binData, n):
    a = list()
    b = list()
    binDatalen = np.arange(1, len(binData), 1)

    st = binData[1]
    for i in binDatalen:
        if st == 1 and binData[i] == 0:
            a.append(i)
        elif st == 0 and binData[i] == 1:
            b.append(i)
        st = binData[i]

    a = np.array(a, dtype=int)
    b = np.array(b, dtype=int)

    if len(a) > n or len(b) > n:
        print(len(a), len(b))
        return -1
    elif len(a) != len(b):
        return 0
    elif len(a) == len(b) and len(a) < n:
        return 0
    elif len(a) == len(b) and len(a) == n:
        return ((a + b) // 2).reshape(-1, 1)

'''
遞迴尋找二值化訊號中心點
'''
def RecursionFineBinSigalCerter(signal, p, n):
    l = np.linspace(0, 1, n)

    for i in l:
        bs = CreatBinSigal(signal, i)
        gp = FindBinSigalCerter(bs, p)

        if type(gp) == type(np.array([])):
            return gp, i
        if gp == -1:
            break
    return -1, None
