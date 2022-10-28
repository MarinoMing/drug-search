##此程序通过设置相似指纹来模拟计算测试数据
from rdkit import Chem
from rdkit import DataStructs
from NNModel import netModule1
from NNModel import netModule2
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mserr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import numpy as np
import math
from sklearn.metrics import mean_absolute_error as mae #MAE



def svmModel1(x_train,y_train,x_val,x_test):
    svm_reg = SVR(kernel='rbf',epsilon=0.1, C=100,max_iter=-1,gamma=0.001)
    svm_reg.fit(x_train, y_train)
    preLabel=svm_reg.predict([x_test])

    preValLabel = svm_reg.predict(x_val)
    preValLabel = list(preValLabel)
    return preLabel, preValLabel

def svmModel2(x_train,y_train,x_val,x_test):
    svm_reg = SVR(kernel='rbf',epsilon=0.01, C=1000,max_iter=-1,gamma=0.001)
    svm_reg.fit(x_train, y_train)
    preLabel=svm_reg.predict([x_test])

    preValLabel = svm_reg.predict(x_val)
    preValLabel = list(preValLabel)

    return preLabel, preValLabel
def randForestModel(x_train,y_train,x_val,x_test):
    # y_train=[[i] for i in y_train]
    clf = RandomForestRegressor()
    clf.fit(x_train,y_train)
    preLabel = clf.predict([x_test])
    preLabel = list(preLabel)

    preValLabel = clf.predict(x_val)
    preValLabel = list(preValLabel)

    return preLabel,preValLabel

def createSimilar(path1,path2):


    df = pd.read_csv(path1, header=0)
    saveSimilar=[]
    for i in tqdm(range(df['SMILES'].size)):
        selfName=df.loc[i,['SMILES']].values[0]
        ms1=Chem.MolFromSmiles(selfName)
        fp1=Chem.RDKFingerprint(ms1)
        tempList=[]
        for j in range(df['SMILES'].size):
            if i<=j:
                ms2 = Chem.MolFromSmiles(df.loc[j,['SMILES']].values[0])
                fp2 = Chem.RDKFingerprint(ms2)
                tempList.append(DataStructs.FingerprintSimilarity(fp1,fp2))
        saveSimilar.append(tempList)
    saveSimilarDataFrame=pd.DataFrame(saveSimilar)
    saveSimilarDataFrame.to_csv(path2,header=0)
# import os
# fileNameList=os.listdir('./SMILES')
#
# for N in fileNameList:
#     path1='./SMILES/'+N
#     path2='./smilarCal/'+'saveSimilarDataFrame_'+N.split('s')[0]+'.csv'
#     createSimilar(path1,path2)

def addSimlar(path1,path2):
    df=pd.read_csv(path1,header=None)
    print(df.shape[0])
    for i in range(1,df.shape[0]):
        keepValues=df.loc[i,].values.tolist()
        keepValues=keepValues[:-i]
        print('i:{}'.format(i))
        for j in range(i):
            keepValues.insert(j,df.loc[j,i])
        df.loc[i,:]=keepValues
    df.to_csv(path2,header=None,index=False)
    # print(df)
# import os
# path1='./smilarCal/saveSimilarDataFrame_DHFR.csv'
# path2='./smilarCal/AllSimilarDataFrame_DHFR.csv'
# addSimlar(path1,path2)

# 计算相关度
def computeCorrelation(x,y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0,len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar**2
        varY += difYYbar**2
    SST = math.sqrt(varX * varY)
    return SSR/SST

def sortSmilar(path1,path2):
    # 此处将摩根分子指纹的相似性比较高的分子对应序号保存在列表中
    df=pd.read_csv(path1,header=None)
    saveSort=[]
    for i in range(df.shape[0]):
        temp=[]
        smilarList=df.loc[i,:].values.tolist()
        sorted_id = sorted(range(len(smilarList)), key=lambda k: smilarList[k], reverse=True)
        for j in range(len(sorted_id)):
            if smilarList[sorted_id[j]]!=1:
                print(j,smilarList[sorted_id[j]])
                temp.append(sorted_id[j])
                # if df.shape[0]==len(temp):
                #     break
        saveSort.append(temp)
    saveSortDataFrame=pd.DataFrame(saveSort)
    saveSortDataFrame.to_csv(path2)
# path1='./smilarCal/AllSimilarDataFrame/AllSimilarDataFrame_DHFR.csv'
# path2='./smilarCal/SortSmilar_DHFR.csv'
# sortSmilar(path1,path2)

def Train(path1,path2,savePath):
    print('开始Train程序')

    nodeDict1 = {'inputNode': 1024, 'hidNode': 56, 'outNode': 1, 'lr': 0.001, 'epochs': 900}
    nodeDict2={'inputNode': 1024, 'hidNode': 102,'hidNode1': 56, 'outNode': 1, 'lr': 0.005, 'epochs': 300}


    dfSmilarSort = pd.read_csv('./smilarCal/sortSmilar/SortSmilar_'+path1+'.csv', index_col=0)
    dfFinger=pd.read_csv('./ECFP/'+path2+'MorganFingerPrint.csv',index_col=0)
    y=[]
    pre=[]
    allCalList=[]
    length=dfFinger.shape[0]
    for i in tqdm(range(length)):
        sortId=dfSmilarSort.loc[i,:].values.tolist()
        valSort =sortId[30:38:2]         #验证集在训练数据中的序号
        valSort=[int(i) for i in valSort]



        x_val=dfFinger.iloc[valSort,:-1].values
        y_val=dfFinger.iloc[valSort,-1].values



        valSort.append(i)

        trainSort=[i for i in range(length) if i not in valSort]


        x_test = dfFinger.iloc[i].iloc[:-1].tolist()
        y_test = [dfFinger.iloc[i].iloc[-1]]

        x_train=dfFinger.iloc[trainSort,:-1].values
        y_train=dfFinger.iloc[trainSort,-1].values
        y_train=[[i] for i in y_train]



        preLabel1,preValLabel1= svmModel1(x_train, y_train,x_val, x_test)
        print('--------------------------------------------------------------------------')
        preLabel2,preValLabel2 = svmModel2(x_train, y_train, x_val,x_test)
        # preLabel2, preValLabel2 = randForestModel(x_train, y_train, x_val, x_test)
        r1=r2_score(y_val,preValLabel1)
        r2=r2_score(y_val,preValLabel2)

        eList1=[i-j for i,j in zip(y_val,preValLabel1)]
        eList2 = [i - j for i, j in zip(y_val, preValLabel1)]
        e1=sum(eList1)/len(eList1)
        e2 = sum(eList2) / len(eList2)
        mse1=np.sqrt(mserr(y_val,preValLabel1))
        mse2 = np.sqrt(mserr(y_val, preValLabel2))

        print(mse1,mse2,e1,e2)
        if mse1<mse2:
            preLabel=preLabel1[0]
        else:
            preLabel=preLabel2[0]


        err=y_test[0]-preLabel
        print(y_test[0],preLabel,preLabel1[0],preLabel2[0],err)
        print('y_val', y_val)
        print('pre_val',preValLabel1,preValLabel2)

        y.append(y_test[0])
        pre.append(preLabel)
        allCalList.append([y_test[0],preLabel,preLabel1[0],preLabel2[0],err])

    print(r2_score(y,pre))
    print('RMSE',np.sqrt(mserr(y,pre)))



    preDataFrame=pd.DataFrame(allCalList,columns=['y','pre','pre1','pre2','error'])
    preDataFrame.to_csv('SVRpreDataFrame'+savePath+'.csv')




if __name__ == '__main__':
    # addSimlar()
    # Train()
    # calR2()
    path1=path2=savePath = 'ER@'
    Train(path1, path2, savePath)
    pass


