'''
A temp file for quick calculation
'''
import numpy as np
from sklearn.preprocessing import Imputer
import algorithms as algo
import pandas as pd

import lib.util.perform as perf

d1 = pd.read_csv("E:/GEODATA/result/untrt_MCF7_base_2000x978.csv")
d2 = pd.read_csv("E:/GEODATA/result/untrt_PC3_base_2000x978.csv")

d1['Label'] = 0
d2['Label'] = 1

Da = pd.concat([d1,d2])
GeneExp=Da.iloc[:,1:-1]
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
          ,'RCA':{}}
          # ,'LFDA':{'k':2, 'dim': 50}  'NCA':{'learning_rate':0.01}'RCA':{ 'num_chunks':150, 'chunk_size':3}

selected = ['GSEA','BASE','LMNN','SDML','LSML','LFDA','NCA']
options = algo.select(methods, selected)

Result = algo.ALGO(GeneExp, Label,  **options)

Dist = Result.Dist
Dist['SiamDen']=np.load('Dist.npy')
# Train = Result.inds_train
# Test = Result.inds_test
Train=np.load('inds_train.npy')
Test=np.load('inds_test.npy')

amin,amax=Dist['SiamDen'].min(),Dist['SiamDen'].max()
Dist['SiamDen']=1.0-(Dist['SiamDen']-amin)/(amax-amin)

perf.roc(Dist, Label, save_figure=True)

for method,dist in Dist.items():
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
