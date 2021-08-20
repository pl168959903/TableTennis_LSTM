from TB_Library import TB_Signal as tb_sg
from TB_Library import TB_Data as tb_dt

from os import path
import numpy as np
import pandas as pd
from tqdm import trange

def TableTennisDataSplit(df, wave_c, detrens_n = 5, lp_order = 2, bin_c = 100, fft_f = 0):

    axis9 = df.loc[df.index, 'ax':'rz'].to_numpy().reshape(-1, 9)

    axisA = df.loc[df.index, 'ax':'az'].to_numpy().reshape(-1, 3)
    axisG = df.loc[df.index, 'gx':'gz'].to_numpy().reshape(-1, 3)

    axisAbsSum_A = np.zeros([len(axisA), 1])
    axisAbsSum_G = np.zeros([len(axisG), 1])

    for i in range(len(axis9)):
        axisAbsSum_A[i] = (abs(axisA[i][0]) + abs(axisA[i][1]) + abs(axisA[i][2]))
        axisAbsSum_G[i] = (abs(axisG[i][0]) + abs(axisG[i][1]) + abs(axisG[i][2]))

    f = axisAbsSum_A + axisAbsSum_G

    f_dt = tb_sg.Detrends(f, detrens_n)

    fftMax = tb_sg.GetFFTMaxValue(f_dt.squeeze())

    if fft_f == 0:
        lowpass_f = fftMax * 4
    else:
        lowpass_f = fft_f 
    
    

    f_dt_low = tb_sg.LowpassFilter(f_dt, lowpass_f, order = lp_order)

    f_dt_low_maxmin = tb_sg.MaxMin(f_dt_low)

    centerData = np.ones([len(f_dt_low_maxmin),1]) - f_dt_low_maxmin
    ct, v = tb_sg.RecursionFineBinSigalCerter(centerData, p = wave_c , n = bin_c)
    if type(ct) == type(1):
        if ct == -1:
            return None

    # cutList = list()

    # for i in np.arange(0,len(ct)-1, 1):
    #     cutList.append((ct[i] + ct[i+1]) // 2)
    # cutList = np.array(cutList)

    cutList = FindWaveWidth(centerData, ct, 3)

    axis9Cut = list()

    for i in range(len(cutList)):
        head = cutList[i][0].squeeze()
        tail = cutList[i][2].squeeze()
        cutArray = axis9[head: tail,:]
        axis9Cut.append(cutArray)

    return axis9Cut

def SplitDataFromPathDf(pathDf, n = 10, paramDf = None):
    pathDfLen = len(pathDf)

    retDf = pd.DataFrame(columns=['path', 'class','wave_c','detrens_n','lowpass_order','bin_c','fft_f'])
    retList = list()

    for i in trange(pathDfLen):
        newDf = pd.DataFrame(columns=['path', 'class', 'wave_c','detrens_n','lowpass_order','bin_c','fft_f'])
        if paramDf is None:
            newDf = newDf.append({  'path' :            pathDf.loc[i,'path'], 
                                    'class' :           pathDf.loc[i,'class'],
                                    'wave_c' :          n, 
                                    'detrens_n' :       5, 
                                    'lowpass_order' :   2, 
                                    'bin_c' :           100, 
                                    'fft_f':            0
                                }, ignore_index=True)
        else:
            newDf = newDf.append({  'path' :            pathDf.loc[i,'path'], 
                                    'class' :           pathDf.loc[i,'class'],
                                    'wave_c' :          pathDf.loc[i,'wave_c'],
                                    'detrens_n' :       pathDf.loc[i,'detrens_n'], 
                                    'lowpass_order' :   pathDf.loc[i,'lowpass_order'], 
                                    'bin_c' :           pathDf.loc[i,'bin_c'], 
                                    'fft_f':            pathDf.loc[i,'fft_f']
                                }, ignore_index=True)

        df = tb_dt.ReadTableTennisDataToDatameta(newDf.loc[0, 'path'])
        axis9Cut = TableTennisDataSplit(df,   wave_c =    int(    newDf.loc[0, 'wave_c']      ),
                                            detrens_n = int(    newDf.loc[0,'detrens_n']    ),
                                            lp_order =  int(    newDf.loc[0,'lowpass_order']),
                                            bin_c =     int(    newDf.loc[0,'bin_c']        ),
                                            fft_f =     float(  newDf.loc[0, 'fft_f']       ))

        if axis9Cut == None:
            print("None data(" + str(i) + ")" + pathDf.loc[i, 'path'])

        retList.append(axis9Cut)
        retDf = retDf.append(newDf, ignore_index = True)

    return retList, retDf

def FixTrainDataList(paramDf , fixIndex, fixDataList):
    ls = fixDataList
    df = tb_dt.ReadTableTennisDataToDatameta(paramDf.loc[fixIndex, 'path'])
    axis9Cut = TableTennisDataSplit(df,   wave_c = int(paramDf.loc[fixIndex, 'wave_c']),
                                            detrens_n = int(paramDf.loc[fixIndex,'detrens_n']),
                                            lp_order = int(paramDf.loc[fixIndex,'lowpass_order']),
                                            bin_c = int(paramDf.loc[fixIndex,'bin_c']),
                                            fft_f = float(paramDf.loc[fixIndex, 'fft_f']))
    ls[fixIndex] = axis9Cut
    return ls


def FindWaveWidth(data_n, TopPointList, startShift):
    retval = list()
    for i in range(len(TopPointList)):
        topPointX = TopPointList[i]

        if (topPointX - startShift) > 0:
            topPointX_n = topPointX - startShift
        else:
            topPointX_n = 0

        if (topPointX + startShift)  < len(data_n):
            topPointX_p = topPointX + startShift
        else:
            topPointX_p = len(data_n) - 1

        while True :
            if topPointX_n == 0 : break
            if data_n[topPointX_n] <= data_n[topPointX_n-1]:
                topPointX_n -= 1
            else:
                break

        while True :
            if topPointX_p == (len(data_n) - 1) : break
            if data_n[topPointX_p] <= data_n[topPointX_p+1]:
                topPointX_p += 1
            else:
                break

        retval.append([topPointX_n, topPointX, topPointX_p])
    return retval
