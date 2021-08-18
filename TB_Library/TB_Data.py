import numpy as np
import pandas as pd
import os
import re

'''
讀取桌球整數TXT檔案到Datameta
'''
def ReadTableTennisDataToDatameta(filename):
    df = pd.read_table(filename, sep=" ", header=None)
    df.drop(df.index[0], inplace=True)
    df.drop(columns=[14, 15], inplace=True)

    df.rename(columns={0: 'ax', 1: 'ay', 2: 'az', 3: 'gx', 4: 'gy', 5: 'gz', 6: 'p1',
              7: 'p2', 8: 'p3', 9: 'p4', 10: 'p5', 11: 'rx', 12: 'ry', 13: 'rz'}, inplace=True)
    df = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'rx',
             'ry', 'rz', 'p1', 'p2', 'p3', 'p4', 'p5']]
    df = df.astype(np.float32)
    df.reset_index(drop=True, inplace=True)
    return df

'''
讀取目錄下按照Class命名的資料夾內所有的檔案
'''
def GetPathAndClassDfFromRoot(rootPath):
    retval = pd.DataFrame()

    allList = os.walk(rootPath)
    pattern = rootPath + '\d'

    retvalRow = 0
    for root, dirs, files in allList:

        if re.match(pattern, root) != None:
            classNumber = float(root.replace(rootPath, ''))

            for fileName in files:
                filePath = root + '/' + fileName
                retval.loc[retvalRow, 'path'] = filePath
                retval.loc[retvalRow, 'class'] = classNumber
                retvalRow = retvalRow + 1

    return retval

'''
輸出CSV
'''
def OutputCSV(df, path):   
    df_SAMPLE = pd.DataFrame.from_dict( df )
    df_SAMPLE.to_csv( path , index=False )

'''
取得相異資料
'''
def GetDfDifference(selfDf, otherDf, columns):
    boolList = selfDf[columns].isin(otherDf[columns])
    for i in range(len(boolList)):
        if boolList[i]:
            selfDf = selfDf.drop(index=i)
    selfDf = selfDf.reset_index(drop = True)
    return selfDf

'''
建立資料夾
'''
def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''
使用路徑取得檔案名稱
'''
def GetFileNameFormPathString(path):
    file_path = os.path.splitext(path)[0]
    file_name = file_path.split('/')[-1]
    return file_name
