from TB_Library import TB_Signal as tb_sg 


import os
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


modelRootPath = './Model/'
trainDataRootPath = './train_data/'
validDataRootPath = './valid_data/'

modelFileName = 'TB_SwingsClassModel.h5'
modelTrainCsvFileName = 'TB_SwingsClassModel_train_list.csv'

############################################################################

def CreatModel(sample, feature, class_n):
    inputs = keras.Input(shape=(sample,feature), dtype="float64")

    layer_1 = layers.Bidirectional(layers.LSTM(32))(inputs)

    outputs = layers.Dense(class_n, activation="sigmoid")(layer_1)

    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def OutPutModelAndTrainList(model, trainDataDf):

    model.save(modelRootPath + modelFileName)

    trainDataDf_sample = pd.DataFrame.from_dict( trainDataDf )
    trainDataDf_sample.to_csv( modelRootPath +  modelTrainCsvFileName, index=False )

def GetModelTrainDf():
    csvFilePath = modelRootPath + modelTrainCsvFileName

    if os.path.isfile(csvFilePath):
        modelTrainDf = pd.read_csv(csvFilePath)
    else:
        modelTrainDf = None
    return modelTrainDf

def GetModel():
    modelFilePath = modelRootPath + modelFileName
    if os.path.isfile(modelFilePath):
        model = keras.models.load_model(modelFilePath)
    else:
        model = None
    return model

def SplitTrainDataAndRandom(xt, yt, per):
    x_train, x_test, y_train, y_test = train_test_split(xt, yt, test_size = per/100)
    print(per/100)
    return [x_train, y_train], [x_test, y_test]

def GetTrainData(dataList, paramDf, dataMaxLen):
    ytrain = list()
    y = paramDf[['class']].to_numpy()
    for i in range(len(y)):
        for j in range(len(dataList[i])):
            ytrain.append(paramDf.loc[i,'class'])
    ytrain = np.array(ytrain).reshape(-1,1)

    ########################################################################
    xtrain = list()
    for i in range(len(dataList)):
        for j in range(len(dataList[i])):
            xtrain.append(dataList[i][j])
    xtrain = keras.preprocessing.sequence.pad_sequences(xtrain, maxlen = dataMaxLen)
    xtrain = AxisDataMaxMin(xtrain)
    ########################################################################

    return xtrain, ytrain

def AxisDataMaxMin(axis):
    axis_T = axis.T
    axis_T = axis_T.astype(np.float64)

    for i in range(len(axis_T)):
        axis_T[i] = tb_sg.MaxMin(axis_T[i])

    return axis_T.T
