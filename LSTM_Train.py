# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import TB_Library.TB_Data as tb_dt
import TB_Library.TB_Model as tb_md
import TB_Library.TB_Split as tb_sp

import pandas as pd


# %%
# 路徑
trainDataRootPath = './train_data/'
validDataRootPath = './valid_data/'


# %%
# 根據資料夾名稱，取出原始資料路徑和類別
folderDf = tb_dt.GetPathAndClassDfFromRoot(trainDataRootPath)


# %%
modelDf = tb_md.GetModelTrainDf()


# %%
if modelDf is None:
    modelDf = pd.DataFrame()
    trainDf = folderDf
else:
    trainDf = tb_dt.GetDfDifference(folderDf, modelDf, 'path')


# %%
if trainDf.empty:
    exit()


# %%
# 分割檔案資料
trainDataList, paramDf = tb_sp.SplitDataFromPathDf(trainDf, 10)


# %%
# 轉換成訓練資料
xtrain , ytrain = tb_md.GetTrainData(trainDataList, paramDf, 200)


# %%
#--------------------------------------------------------------------------------------
# 模型建立


# %%
model = tb_md.GetModel()
if model is None:
    model = tb_md.CreatModel(sample=200,feature=9,class_n=8)


# %%
trainData, valData = tb_md.SplitTrainDataAndRandom(xtrain, ytrain, 20)


# %%
history = model.fit(trainData[0], trainData[1], batch_size=32, epochs=200, validation_data=(valData[0], valData[1]))


# %%
# index = 220

# testx = np.asarray([xtrain[index]])
# testy = ytrain[index]

# print("NO.%d Predict Result is Class " % index, end = '')
# print(tf.math.argmax(model.predict(testx)[0]).numpy())

# print("Compare with true result. The true result is %d" % int(testy))


# %%
modelDf = modelDf.append(trainDf)
tb_md.OutPutModelAndTrainList(model, modelDf)


