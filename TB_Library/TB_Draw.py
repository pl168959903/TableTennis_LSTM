import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from TB_Library import TB_Data as tb_dt


from tqdm import tqdm

train_data_class_list_path = './train_data/'
valid_data_class_list_path = './valid_data/'

'''
建立新畫布
'''
def FigNew(l=15, w=3):
    plt.figure(figsize=(l, w))
    plt.xlabel('Time(s)')
    plt.ylabel('Value')
    plt.grid()
    FigAddSignal.data_c = 0

'''
為畫布添加訊號
'''
def FigAddSignal(data, labelName='', fz=85, showAfterAdd=False):
    sampleSize = len(data)
    dataTimes = sampleSize / fz
    dataTimeLines = np.linspace(0, dataTimes, sampleSize)

    if labelName == '':
        labelName = 'signal' + str(FigAddSignal.data_c)
        FigAddSignal.data_c += 1

    plt.plot(dataTimeLines, data, label=labelName)
    plt.legend(loc='upper left')

    if showAfterAdd:
        plt.show()
FigAddSignal.data_c = 0

'''
九軸 Numnp 圖畫
'''
def FigAddAxis(axisnp, pf=None,  l=15, w=2, showAfterAdd=False):
    np_list = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'RX', 'RY', 'RZ']

    if type(pf) == type(1):
        FigAddSignal(axisnp.T[pf], labelName=np_list[pf], showAfterAdd=False)

    if type(pf) == type(''):

        if pf == 'ALL':
            for i in range(np_list.index('AX'), np_list.index('RZ')+1):
                FigAddSignal(
                    axisnp.T[i], labelName=np_list[i], showAfterAdd=False)

        elif pf == "A":
            for i in range(np_list.index('AX'), np_list.index('AZ')+1):
                FigAddSignal(
                    axisnp.T[i], labelName=np_list[i], showAfterAdd=False)

        elif pf == "G":
            for i in range(np_list.index('GX'), np_list.index('GZ')+1):
                FigAddSignal(
                    axisnp.T[i], labelName=np_list[i], showAfterAdd=False)

        elif pf == "R":
            for i in range(np_list.index('RX'), np_list.index('RZ')+1):
                FigAddSignal(
                    axisnp.T[i], labelName=np_list[i], showAfterAdd=False)

        else:
            FigAddSignal(axisnp.T[np_list.index(
                pf)], labelName=np_list[np_list.index(pf)], showAfterAdd=False)

    if type(pf) == type([]):
        for i in pf:
            if type(i) == type(1):
                FigAddSignal(
                    axisnp.T[i], labelName=np_list[i], showAfterAdd=False)
            if type(i) == type(''):
                FigAddSignal(axisnp.T[np_list.index(
                    i)], labelName=np_list[np_list.index(i)], showAfterAdd=False)

    if pf == None:
        for i in range(len(np_list)):
            FigAddSignal(
                axisnp.T[i], labelName=np_list[i], showAfterAdd=False)

    if showAfterAdd == True:
        plt.show()

'''
儲存Axis繪圖
'''
def SaveImageFromAxis(axis, path):
    FigNew(l = 5)
    FigAddAxis(axis, pf='ALL')
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()

'''
在跟目錄建立資料夾並儲存所有繪圖
'''
def SaveImagePathRoot(dataList, dataParam):
    
    saveImageRoot = './Data_Image/'
    if os.path.exists(saveImageRoot):
        shutil.rmtree(saveImageRoot)

    listSize = len(dataList)

    tailObjects = 0
    for i in range(listSize):
        for k in dataList[i]:
            tailObjects += 1

    progress = tqdm(total=tailObjects)

    for i in range(listSize):
        obj = dataList[i]

        objPath  = dataParam.loc[i, 'path']
        objClass = int(dataParam.loc[i, 'class'])
        objfileName = tb_dt.GetFileNameFormPathString(objPath)

        savePathRoot = saveImageRoot + str(objClass) + '/'

        for j in range(len(obj)):
            cell = obj[j]
            tb_dt.CreateDir(savePathRoot)
            SaveImageFromAxis(cell, savePathRoot + objfileName + '_' + str(i) + '_' + str(j) + '.png')
            progress.update(1)

