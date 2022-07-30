# -*-coding:utf-8-*-
'''
这部分代码是修改后的，加入了xgboost模型，以及交叉验证方法
'''
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差

pathName=['ace.xls','ache.xls','bzr.xls','COX2.xls','DHFR.xls']
path = './completeMaccs/'+pathName[4]
groupNum = 5  # 将所有数据分成5份，20%作为测试，剩下全部用于训练
col = []
for i in range(166):
    col.append(i)
inputData = pd.read_excel(path, header=None, usecols=col)
labelData = pd.read_excel(path, header=None, usecols=[166])

inputDataList = inputData.values.tolist()
labelDataList = labelData.values.tolist()

dataLength = len(labelDataList)

member = dataLength // groupNum
remain = dataLength % groupNum

G1 = []
G2 = []
G3 = []
G4 = []
Gtest = []
G1Label = []
G2Label = []
G3Label = []
G4Label = []
GtestLabel = []


seedNum=10
iterations=900
sort = [0, 1, 2, 3, 4]

np.random.seed(seedNum)
np.random.shuffle(sort)
print(sort)
for i in range(member):
    # np.random.shuffle(sort)
    G1.append(inputDataList[groupNum * i + sort[0]])
    G1Label.append(labelDataList[groupNum * i + sort[0]])
    G2.append(inputDataList[groupNum * i + sort[1]])
    G2Label.append(labelDataList[groupNum * i + sort[1]])
    G3.append(inputDataList[groupNum * i + sort[2]])
    G3Label.append(labelDataList[groupNum * i + sort[2]])
    G4.append(inputDataList[groupNum * i + sort[3]])
    G4Label.append(labelDataList[groupNum * i + sort[3]])
    Gtest.append(inputDataList[groupNum * i + sort[4]])
    GtestLabel.append(labelDataList[groupNum * i + sort[4]])


nameData=[G1,G2,G3,G4,Gtest]
nameLabel=[G1Label,G2Label,G3Label,G4Label,GtestLabel]



if remain!=0:
    random.seed(seedNum)
    remainNum=random.sample(range(0,5),remain)
    t=0
    for k in remainNum:
        nameData[k].append(inputDataList[groupNum * groupNum + t])
        nameLabel[k].append(labelDataList[groupNum * groupNum + t])
        t+=1

#style 1
G=G1+G2+G3+G4
GLabel=G1Label+G2Label+G3Label+G4Label
GLabelLi=[L[0] for L in GLabel]
GtestLabelLi=[L[0] for L in GtestLabel]
X_train=np.array(G)
y_train=np.array(GLabelLi)
X_test=np.array(Gtest)
y_test=np.array(GtestLabelLi)

#style 2
# G=G1+G2+G3+Gtest
# GLabel=G1Label+G2Label+G3Label+GtestLabel
# GLabelLi=[L[0] for L in GLabel]
# GtestLabelLi=[L[0] for L in G4Label]
# X_train=np.array(G)
# y_train=np.array(GLabelLi)
# X_test=np.array(G4)
# y_test=np.array(GtestLabelLi)

#style 3
# G=G1+G2+G4+Gtest
# GLabel=G1Label+G2Label+G4Label+GtestLabel
# GLabelLi=[L[0] for L in GLabel]
# GtestLabelLi=[L[0] for L in G3Label]
# X_train=np.array(G)
# y_train=np.array(GLabelLi)
# X_test=np.array(G3)
# y_test=np.array(GtestLabelLi)

#style 4
# G=G1+G3+G4+Gtest
# GLabel=G1Label+G3Label+G4Label+GtestLabel
# GLabelLi=[L[0] for L in GLabel]
# GtestLabelLi=[L[0] for L in G2Label]
# X_train=np.array(G)
# y_train=np.array(GLabelLi)
# X_test=np.array(G2)
# y_test=np.array(GtestLabelLi)

#style 5
# G=G2+G3+G4+Gtest
# GLabel=G2Label+G3Label+G4Label+GtestLabel
# GLabelLi=[L[0] for L in GLabel]
# GtestLabelLi=[L[0] for L in G1Label]
# X_train=np.array(G)
# y_train=np.array(GLabelLi)
# X_test=np.array(G1)
# y_test=np.array(GtestLabelLi)

class Net(nn.Module):
    def __init__(self,inputNode,hidNode,hidNode1,hidNode2,outNode):
        super(Net,self).__init__()
        self.hidLayer=nn.Linear(inputNode,hidNode)
        self.hidLayer1 = nn.Linear(hidNode, hidNode1)
        self.hidLayer2 = nn.Linear(hidNode1, hidNode2)
        self.outLayer=nn.Linear(hidNode2,outNode)

    def forward(self,x):
        x=F.relu(self.hidLayer(x))
        x=F.relu(self.hidLayer1(x))
        x = F.relu(self.hidLayer2(x))
        out=self.outLayer(x)
        return out

device = torch.device("cuda:0")
def train(G,Glabel,Gver,Gverlabel,learnEff,inter,inputNode,hidNode,hidNode1,hidNode2,outNode):
    model = Net(inputNode, hidNode, hidNode1,hidNode2, outNode)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learnEff)
    GTensor=torch.Tensor(G)
    GlabelTensor=torch.Tensor(Glabel)

    GverTensor = torch.Tensor(Gver)
    gpu_GverTensor = GverTensor.to(device)

    gpu_GTensor=GTensor.to(device)
    gpu_GlabelTensor=GlabelTensor.to(device)
    GverList=[L[0] for L in Gverlabel]
    # GverList_arry=np.array(GverList)
    for i in range(inter):
        predict = model(gpu_GTensor)
        loss = criterion(predict, gpu_GlabelTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pre_ver = model(gpu_GverTensor)
        pre_ver_list = torch.Tensor.cpu(pre_ver.reshape(-1)).tolist()
        mean_ver=r2_score(GverList,pre_ver_list)
        # print(i,mean_ver)
        # if mean_ver>0.69:
        #     print('i',i)
        #     print(mean_ver)
        #     break
    meansqu=[i-j for i,j in zip(GverList,pre_ver_list)]
    means=sum(meansqu)/(len(meansqu))
    print(loss)
    return model,means

#style 1
model1,means1=train(G1+G2+G3,G1Label+G2Label+G3Label,G4,G4Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
model2,means2=train(G1+G2+G4,G1Label+G2Label+G4Label,G3,G3Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
model3,means3=train(G1+G3+G4,G1Label+G3Label+G4Label,G2,G2Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
model4,means4=train(G2+G3+G4,G2Label+G3Label+G4Label,G1,G1Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)

#style 2
# model1,means1=train(G1+G2+G3,G1Label+G2Label+G3Label,Gtest,GtestLabel,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model2,means2=train(G1+G2+Gtest,G1Label+G2Label+GtestLabel,G3,G3Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model3,means3=train(G1+G3+Gtest,G1Label+G3Label+GtestLabel,G2,G2Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model4,means4=train(G2+G3+Gtest,G2Label+G3Label+GtestLabel,G1,G1Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)


#style 3
# model1,means1=train(G1+G2+G4,G1Label+G2Label+G4Label,Gtest,GtestLabel,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model2,means2=train(G1+G2+Gtest,G1Label+G2Label+GtestLabel,G4,G4Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model3,means3=train(G1+G4+Gtest,G1Label+G4Label+GtestLabel,G2,G2Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model4,means4=train(G2+G4+Gtest,G2Label+G4Label+GtestLabel,G1,G1Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)

#style 4
# model1,means1=train(G1+G3+G4,G1Label+G3Label+G4Label,Gtest,GtestLabel,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model2,means2=train(G1+G3+Gtest,G1Label+G3Label+GtestLabel,G4,G4Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model3,means3=train(G1+G4+Gtest,G1Label+G4Label+GtestLabel,G3,G3Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model4,means4=train(G3+G4+Gtest,G3Label+G4Label+GtestLabel,G1,G1Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)

#style 5
# model1,means1=train(G2+G3+G4,G2Label+G3Label+G4Label,Gtest,GtestLabel,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model2,means2=train(G2+G3+Gtest,G2Label+G3Label+GtestLabel,G4,G4Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model3,means3=train(G2+G4+Gtest,G2Label+G4Label+GtestLabel,G3,G3Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
# model4,means4=train(G3+G4+Gtest,G3Label+G4Label+GtestLabel,G2,G1Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)



model5,means5=train(G1+G2,G1Label+G2Label,G3+G4,G3Label+G4Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
model6,means6=train(G3+G4,G3Label+G4Label,G1+G2,G1Label+G2Label,learnEff=0.01,inter=iterations,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)



#style 1
GtestTensor=torch.Tensor(Gtest)

#style 2
# GtestTensor=torch.Tensor(G4)
#style 3
# GtestTensor=torch.Tensor(G3)
#style 4
# GtestTensor=torch.Tensor(G2)
#style5
# GtestTensor=torch.Tensor(G1)

gpu_GtestTensor=GtestTensor.to(device)

pre_model1=model1(gpu_GtestTensor)
pre_model2=model2(gpu_GtestTensor)
pre_model3=model3(gpu_GtestTensor)
pre_model4=model4(gpu_GtestTensor)

pre_model5=model5(gpu_GtestTensor)
pre_model6=model6(gpu_GtestTensor)

"""
计算BPC模型训练集的值
"""
G1TrainTensor=torch.Tensor(G1)
gpu_G1TrainTensor=G1TrainTensor.to(device)
G2TrainTensor=torch.Tensor(G2)
gpu_G2TrainTensor=G2TrainTensor.to(device)
G3TrainTensor=torch.Tensor(G3)
gpu_G3TrainTensor=G3TrainTensor.to(device)

#style 1
G4TrainTensor=torch.Tensor(G4)
#style 2
# G4TrainTensor=torch.Tensor(Gtest)
#style 3
# G3TrainTensor=torch.Tensor(Gtest)
#style 4
# G2Tensor=torch.Tensor(Gtest)
#style 5
# G1Tensor=torch.Tensor(Gtest)


gpu_G4TrainTensor=G4TrainTensor.to(device)

pre_G4=model1(gpu_G4TrainTensor)
pre_G3=model2(gpu_G3TrainTensor)
pre_G2=model3(gpu_G2TrainTensor)
pre_G1=model4(gpu_G1TrainTensor)

pre_G1 = torch.Tensor.cpu(pre_G1.reshape(-1)).tolist()
pre_G2 = torch.Tensor.cpu(pre_G2.reshape(-1)).tolist()
pre_G3 = torch.Tensor.cpu(pre_G3.reshape(-1)).tolist()
pre_G4 = torch.Tensor.cpu(pre_G4.reshape(-1)).tolist()

BpcResult=pre_G1+pre_G2+pre_G3+pre_G4

BpcSaveList=[]
BpcSaveList.append(GLabelLi)
BpcSaveList.append(BpcResult)
BpcSaveList=list(map(list,zip(*BpcSaveList)))
BpcTrainSaveData=pd.DataFrame(BpcSaveList)
BpcTrainSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\BpcTrain.xls',header=None,index=False)

"""
end
"""
pre_model1_list = torch.Tensor.cpu(pre_model1.reshape(-1)).tolist()
pre_model2_list = torch.Tensor.cpu(pre_model2.reshape(-1)).tolist()
pre_model3_list = torch.Tensor.cpu(pre_model3.reshape(-1)).tolist()
pre_model4_list = torch.Tensor.cpu(pre_model4.reshape(-1)).tolist()

pre_model5_list = torch.Tensor.cpu(pre_model5.reshape(-1)).tolist()
pre_model6_list = torch.Tensor.cpu(pre_model6.reshape(-1)).tolist()

pre_combine_before=[(i+j+k+p)/4 for i,j,k,p in zip(pre_model1_list,pre_model2_list,pre_model3_list,pre_model4_list)]

y_label_before=[L[0] for L in GtestLabel]   #style 1 G1+G2+G3+G4
# y_label_before=[L[0] for L in G4Label]  #style 2  G1+G2+G3+Gtest
# y_label_before=[L[0] for L in G3Label]      #style 3
# y_label_before=[L[0] for L in G2Label]  #style 4
# y_label_before=[L[0] for L in G1Label]  #style 5

r2_before=r2_score(y_label_before,pre_combine_before)
# print('bpR2__before',r2_before)


pre_model1_list = [i+means1 for i in pre_model1_list]
pre_model2_list = [i+means2 for i in pre_model2_list]
pre_model3_list = [i+means3 for i in pre_model3_list]
pre_model4_list = [i+means4 for i in pre_model4_list]

pre_combine=[(i+j+k+p)/4 for i,j,k,p in zip(pre_model1_list,pre_model2_list,pre_model3_list,pre_model4_list)]

y_label=[L[0] for L in GtestLabel]  #style 1 G1+G2+G3+G4
# y_label=[L[0] for L in G4Label]     #style 2  G1+G2+G3+Gtest
# y_label=[L[0] for L in G3Label]         #style 3
# y_label=[L[0] for L in G2Label]     #style 4
# y_label=[L[0] for L in G1Label]     #style 5

'''
此处计算BPC训练集的决定系数
'''
Gtrain=G1+G2+G3+G4  #style 1
GLabeltrain=G1Label+G2Label+G3Label+G4Label

# Gtrain=G1+G2+G3+Gtest #style 2
# GLabeltrain=G1Label+G2Label+G3Label+GtestLabel

# Gtrain=G1+G2+G4+Gtest #style 3
# GLabeltrain=G1Label+G2Label+G4Label+GtestLabel

# Gtrain=G1+G3+G4+Gtest #style 4
# GLabeltrain=G1Label+G3Label+G4Label+GtestLabel

# Gtrain=G2+G3+G4+Gtest #style 5
# GLabeltrain=G2Label+G3Label+G4Label+GtestLabel


GtrainTensor=torch.Tensor(Gtrain)
gpu_GtrainTensor=GtrainTensor.to(device)

pre_train1=model1(gpu_GtrainTensor)
pre_train2=model2(gpu_GtrainTensor)
pre_train3=model3(gpu_GtrainTensor)
pre_train4=model4(gpu_GtrainTensor)

pre_train1_list = torch.Tensor.cpu(pre_train1.reshape(-1)).tolist()
pre_train2_list = torch.Tensor.cpu(pre_train2.reshape(-1)).tolist()
pre_train3_list = torch.Tensor.cpu(pre_train3.reshape(-1)).tolist()
pre_train4_list = torch.Tensor.cpu(pre_train4.reshape(-1)).tolist()

pre_bpc_train=[(i+j+k+p)/4 for i,j,k,p in zip(pre_train1_list,pre_train2_list,pre_train3_list,pre_train4_list)]
bpr2=r2_score(GLabeltrain,pre_bpc_train)
print('BPCTrain',bpr2)
'''
end
'''


bpr2=r2_score(y_label,pre_combine)
print('BPC',bpr2)
# print('y_label',y_label)
print('BPC预测值',pre_combine)

'''
将BPC预测值保存到excel表中
'''
# BPCSaveList=[]      #
# BPCSaveList.append(y_label)
# BPCSaveList.append(pre_combine)
# BPCSaveData=pd.DataFrame(BPCSaveList)
# BPCSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\BPC.xls',header=None,index=False)
'''
end
'''


print('测试集mse',mean_squared_error(y_label,pre_combine))

pre_combine_cheng=[pow(i*j*k*p,1/4) for i,j,k,p in zip(pre_model1_list,pre_model2_list,pre_model3_list,pre_model4_list)]

r2_cheng=r2_score(y_label,pre_combine_cheng)
print('bpchengR2',r2_cheng)

def svmModel(X_train,y_train):
    svm_reg = SVR(kernel='rbf',epsilon=0.1, C=1,max_iter=-1,gamma=0.05)
    svm_reg.fit(X_train, y_train)
    return svm_reg



svm_reg=svmModel(X_train,y_train)
pre_svm=svm_reg.predict(X_test)

'''
计算SVR模型的训练集预测值
'''
preSvrTrain=svm_reg.predict(X_train)
SvrTrainSaveList=[]
SvrTrainSaveList.append(GLabelLi)
SvrTrainSaveList.append(list(preSvrTrain))
SvrTrainSaveList=list(map(list,zip(*SvrTrainSaveList)))
SvrTrainSaveData=pd.DataFrame(SvrTrainSaveList)
SvrTrainSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\SvrTrain.xls',header=None,index=False)
'''
end
'''
'''计算SVR模型决定系数'''
pre_svmTrain=svm_reg.predict(X_train)
svmR2Train=r2_score(y_train,pre_svmTrain)
print('SVRTrain',svmR2Train)
'''end'''

svmR2=r2_score(y_test,pre_svm)
print('SVR',svmR2)
print('SVR预测值',pre_svm)
print('SVR模型的RMSE',np.sqrt(mean_squared_error(y_test,pre_svm)))
SVRRsultData=[]
SVRRsultData.append(svmR2Train)
SVRRsultData.append(svmR2)
SVRRsultData.append(np.sqrt(mean_squared_error(y_test,pre_svm)))

'''
将SVR预测值保存到excel表中
'''
# SVRSaveList=[]
# SVRSaveList.append(y_label)
# SVRSaveList.append(list(pre_svm))
# SVRSaveData=pd.DataFrame(SVRSaveList)
# SVRSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\SVR.xls',header=None,index=False)
'''
end
'''

def train1(G,Glabel,learnEff,inter,inputNode,hidNode,hidNode1,hidNode2,outNode):
    lossList=[]
    model = Net(inputNode, hidNode, hidNode1,hidNode2, outNode)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learnEff)
    GTensor=torch.Tensor(G)
    GlabelTensor=torch.Tensor(Glabel)


    gpu_GTensor=GTensor.to(device)
    gpu_GlabelTensor=GlabelTensor.to(device)
    for i in range(inter):
        predict = model(gpu_GTensor)
        loss = criterion(predict, gpu_GlabelTensor)
        optimizer.zero_grad()
        lossList.append(loss/len(G))
        loss.backward()
        optimizer.step()
    print(loss)
    return model,lossList
import time
startTime=time.time()
model1234,lossList=train1(G,GLabel,learnEff=0.01,inter=850,inputNode=166,hidNode=100,hidNode1=60,hidNode2=20,outNode=1)
endtime=time.time()
print('收敛时间',endtime-startTime)

GtestTensor=torch.Tensor(Gtest) #style 1 G(1,2,3,4)
# GtestTensor=torch.Tensor(G4)    #style 2 G(1,2,3,test)
# GtestTensor=torch.Tensor(G3)    #style 3
# GtestTensor=torch.Tensor(G2)    #style 4
# GtestTensor=torch.Tensor(G1)    #style 5

gpu_GtestTensor=GtestTensor.to(device)

# pre_model1234=model1234(gpu_GtestTensor)
# pre_model1234_list = torch.Tensor.cpu(pre_model1234.reshape(-1)).tolist()
# r1234=r2_score(y_label,pre_model1234_list)
# print("r1234",r1234)

# pre_svm_bp1_4=[(i+j)/2 for i,j in zip(pre_svm,pre_model1234_list)]
pre_svm_bpCobine=[(i+j)/2 for i,j in zip(pre_svm,pre_combine)]

# r2_svm_bp1_4=r2_score(y_label,pre_svm_bp1_4)
r2_svm_bpCobine=r2_score(y_label,pre_svm_bpCobine)
print('BPCSVR预测值',pre_svm_bpCobine)
print('BPCSVR',r2_svm_bpCobine)

#计算均方根误差
print("BPCSVR模型的均方根误差",np.sqrt(mean_squared_error(y_label,pre_svm_bpCobine)))
# print('r2_svm_bp1_4',r2_svm_bp1_4)

'''计算BPCSVR模型的决定系数'''
pre_BPCSVR_Train=[(i+j)/2 for i,j in zip(pre_bpc_train,pre_svmTrain)]
r2BpcTrain=r2_score(GLabeltrain,pre_BPCSVR_Train)
print('BPCSVRTrain',r2BpcTrain)
'''end'''

'''
将BPCSVR预测值保存到excel表中
'''
BPCSVRSaveList=[]
BPCSVRSaveList.append(y_label)
BPCSVRSaveList.append(pre_svm_bpCobine)
BPCSVRSaveData=pd.DataFrame(BPCSVRSaveList)
BPCSVRSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\BPCSVR.xls',header=None,index=False)
'''
end
'''

BPCSVRRsultData=[]
BPCSVRRsultData.append(r2BpcTrain)
BPCSVRRsultData.append(r2_svm_bpCobine)
BPCSVRRsultData.append(np.sqrt(mean_squared_error(y_label,pre_svm_bpCobine)))


pre_combine_TWO=[(i+j)/2 for i,j in zip(pre_model5_list,pre_model6_list)]

r2_TWO=r2_score(y_label_before,pre_combine_TWO)
print('bpR2__TWO',r2_TWO)


# r2save=[r2_before,bpr2,r2_cheng,svmR2,r1234,r2_svm_bp1_4,r2_svm_bpCobine,r2_TWO]
# L1=len(y_label)
# L2=len(r2save)
# L=L1-L2
# for i in range(L):
#     r2save.append(0)
#
# nameList=path.split('/')
# nameStr=nameList[-1].split('.')
# Filename=nameStr[0]

# saveFilePath='C://Users//Administrator//Desktop//新建文件夹//'+str(seedNum)+str(Filename)+'result.xls'
# df=pd.DataFrame({'label': y_label,'BPcombine': pre_combine,'BP1234':pre_svm_bp1_4,'svr':pre_svm,'svm_bpCombine':pre_svm_bpCobine,'r2save':r2save})
# df.to_excel(saveFilePath,index=False)


#决策树
from sklearn import tree
from sklearn.tree import export_graphviz
def model1_regressorTree(X_train,y_train,X_test,y_test):
    re=tree.DecisionTreeRegressor(random_state=0
                                  ,max_depth=4
                                  ,min_samples_leaf=3

                                  )
    re.fit(X_train,y_train)
    y_p=re.predict(X_test)
    y_pL=list(y_p)
    print("DT预测值",y_pL)
    print('DT', r2_score( y_test,y_p))
    print('DT模型的RMSE',np.sqrt(mean_squared_error(y_test,y_p)))

    pre_train = re.predict(X_train)
    print('DTTrain', r2_score(y_train, pre_train))
    DTRsultData = []
    DTRsultData.append(r2_score(y_train, pre_train))
    DTRsultData.append(r2_score( y_test,y_p))
    DTRsultData.append(np.sqrt(mean_squared_error(y_test,y_p)))

    ##导出dot文件
    export_graphviz(
        re,
        out_file=r"C:\Users\Administrator\Desktop\honor_tree_re.dot",

        rounded=True,
        filled=True
    )

    '''
    将DT预测值保存到excel表中
    '''
    DTSVRSaveList = []
    DTSVRSaveList.append(y_label)
    DTSVRSaveList.append(y_pL)
    DTSVRSaveData = pd.DataFrame(DTSVRSaveList)
    DTSVRSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\DT.xls', header=None, index=False)
    '''
    end
    '''
    return y_p,DTRsultData

pre_tree,DTRsultData=model1_regressorTree(X_train,y_train,X_test,y_test)


#最临近回归
from sklearn.neighbors import KNeighborsRegressor
def model3_knsFun(X_train,y_train,X_test,y_test):
    kns=KNeighborsRegressor()
    kns.fit(X_train,y_train)
    y_kns=kns.predict(X_test)
    y_knsL=list(y_kns)
    print("KNN预测值：",y_knsL)
    print('KNN模型的R2', r2_score( y_test,y_kns))
    print('KNN模型的RMSE',np.sqrt(mean_squared_error(y_test,y_kns)))

    pre_train = kns.predict(X_train)
    print('KNNTrain的R2', r2_score(y_train, pre_train))

    KNNRsultData = []
    KNNRsultData.append(r2_score(y_train, pre_train))
    KNNRsultData.append(r2_score( y_test,y_kns))
    KNNRsultData.append(np.sqrt(mean_squared_error(y_test,y_kns)))

    '''
        将KNN预测值保存到excel表中
    '''
    DTSVRSaveList = []
    DTSVRSaveList.append(y_label)
    DTSVRSaveList.append(y_knsL)
    DTSVRSaveData = pd.DataFrame(DTSVRSaveList)
    DTSVRSaveData.to_excel('C:\\Users\\Administrator\\Desktop\\excel\\KNN.xls', header=None, index=False)
    '''
    end
    '''
    return y_kns,KNNRsultData
pre_kns,KNNRsultData=model3_knsFun(X_train,y_train,X_test,y_test)

finall=[(i+j+k+p)/4 for i,j,k,p in zip(pre_combine,pre_svm,pre_kns,pre_tree)]
r2_fina=r2_score( y_test,finall)
print('r2_finall',r2_fina)

svr_tree=[(i+j)/2 for i ,j in zip(pre_svm,pre_tree)]
print('r2_svr_tree',r2_score(y_test,svr_tree))

svr_kns=[(i+j)/2 for i ,j in zip(pre_svm,pre_kns)]
print('r2_svr_kns',r2_score(y_test,svr_kns))

tree_kns=[(i+j)/2 for i ,j in zip(pre_tree,pre_kns)]
print('r2_tree_kns',r2_score(y_test,tree_kns))

tree_bpcross=[(i+j)/2 for i ,j in zip(pre_tree,pre_svm_bpCobine)]
print('r2_tree_bpcross',r2_score(y_test,tree_bpcross))




#XGBoost模型
from xgboost import XGBRegressor,plot_tree
(X_train,y_train,X_test,y_test)
XGBModel=XGBRegressor(max_depth=10,learning_rate=0.1,objective='reg:gamma')
XGBModel.fit(X_train,y_train)
y_pre_XGBM=XGBModel.predict(X_test)
y_pre_XGBMTrain=XGBModel.predict(X_train)
print(y_pre_XGBM)
print('y_test',y_test)

y_testXGB = np.array(y_test)
y_testXGB = y_testXGB.tolist()
print('y_testXGB',y_testXGB)
print("XGB模型Train的R2",r2_score(y_train,y_pre_XGBMTrain))
print("XGB模型test的R2",r2_score(y_testXGB,y_pre_XGBM))
print("XGB模型的RMSE",np.sqrt(mean_squared_error(y_testXGB,y_pre_XGBM)))

XGBRsultData = []
XGBRsultData.append(r2_score(y_train,y_pre_XGBMTrain))
XGBRsultData.append(r2_score(y_testXGB,y_pre_XGBM))
XGBRsultData.append(np.sqrt(mean_squared_error(y_testXGB,y_pre_XGBM)))

#画图
# import matplotlib.pyplot as plt
# x=range(0,200)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.plot(x,lossList[0:200])
# plt.title('The training loss of compound DHFR in the BP model',fontdict={'weight':'normal','size':20})
# plt.xlabel('The epoches',fontdict={'weight':'normal','size':20})
# plt.ylabel('The loss of training',fontdict={'weight':'normal','size':20})
# plt.show()
# plt.savefig(r'C:\Users\Administrator\Desktop\fig.svg',format='svg',dpi=200)

print('BPCSVR',BPCSVRRsultData)
print('DT',DTRsultData)
print('KNN',KNNRsultData)
print('SVR',SVRRsultData)
print('XGB',XGBRsultData)