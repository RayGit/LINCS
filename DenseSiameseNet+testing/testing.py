'''
A temp file for quick calculation
'''
import numpy as np

import algorithms as algo
import pandas as pd

import lib.util.perform as perf

d1 = pd.read_csv("1x12328.csv")
d2 = pd.read_csv("1x12328.csv")

Da = pd.concat([d1,d2])
print('Da：\n',Da)
GeneExp=Da.iloc[:,1:-1]
print('GeneExp：\n',GeneExp)
GeneExp.reset_index(drop=True)
DataL=Da
Label=DataL['Label']

# options for prepared algorithms
methods = {'BASE':{'metric':'euclidean'}
          ,'GSEA':{'distance':True, 'verbose':True}
          ,'LMNN':{'k':5, 'learn_rate':1e-6, 'regularization':0.7, 'max_iter':500, 'verbose':True}
          ,'ITML':{'num_constraints': 2000,'gamma':20.0}
          ,'SDML':{'balance_param':0.5, 'sparsity_param':0.1}
          ,'LSML':{}
          ,'NCA':{}
          ,'LFDA':{}
          ,'RCA':{'num_chunks':150, 'chunk_size':3}}
          # ,'LFDA':{'k':2, 'dim': 50}  'NCA':{'learning_rate':0.01}

selected = ['BASE']  #选择需要比对的算法
#selected = ['BASE','LMNN','SDML','LSML','LFDA','RCA']  #选择需要比对的算法

options = algo.select(methods, selected)

Result = algo.ALGO(GeneExp, Label,  **options)  #传入algorithms.py 先分配训练样本和测试样本，然后运行相关算法得到对应距离

Dist = Result.Dist
# Dist={}
Dist['SiamDen'] = np.load('Dist_12328.npy')
Train = np.load('inds_train.npy') #训练样本下标
Test = np.load('inds_train.npy')
#在这里加入其他算法的名字和距离矩阵如 Dist['SD']= [array([[]])
amin, amax = Dist['SiamDen'].min(), Dist['SiamDen'].max() # 求最大最小值
Dist['SiamDen'] = 1.0 - (Dist['SiamDen']-amin)/(amax-amin)


perf.roc(Dist, Label, save_figure=True)

for method,dist in Dist.items():
    print(method)
    Predict = perf.knn(dist, Label, Train)
    print (perf.accuracy(Label[Train], Predict[Train]))
    print (perf.accuracy(Label[Test], Predict[Test]))

import matplotlib.pyplot as plt
for method,dist in Dist.items():
    plt.figure(method)
    plt.imshow(dist)
    plt.gca().invert_yaxis()
plt.show()

print("Pipline Finished!")
