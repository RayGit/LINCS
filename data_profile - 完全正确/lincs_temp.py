'''
A temp file for quick calculation
'''

import numpy as np
import pandas as pd
import algorithms as algo

import lib.util.perform as perf

d1 = pd.read_csv("E:/GEO DATA/result/untrt_MCF7_base_2000x978.csv")
d2 = pd.read_csv("E:/GEO DATA/result/untrt_PC3_base_2000x978.csv")

d1['Label'] = 0
d1['Tag'] ='MCF7'

d2['Label'] = 1
d2['Tag'] = 'PC3'
# print(d1.head())
# print(d2.head())
Da = pd.concat([d1,d2])
GeneExp=Da.iloc[:,:]
DataT=Da
DataL=Da
Label=DataL['Label']
Tag=DataT['Tag']

# options for prepared algorithms
methods = {'BASE':{'metric':'euclidean'}
          ,'GSEA':{'distance':True, 'verbose':True}
          ,'LMNN':{'k':5, 'learn_rate':1e-6, 'regularization':0.7, 'max_iter':500, 'verbose':True}
          ,'ITML':{'num_constraints': 2000,'gamma':20.0}
          ,'SDML':{'balance_param':0.5, 'sparsity_param':0.1}
          ,'LSML':{}
          ,'NCA':{'learn_rate':0.01}
          ,'LFDA':{'k':2, 'dim': 50}
          ,'RCA':{'num_chunks':150, 'chunk_size':3}}

selected = ['BASE','LMNN','GSEA','ITML','SDML','NCA','LFDA','RCA']
options = algo.select(methods, selected)

Result = algo.ALGO(GeneExp, Label,Tag)

Dist = Result.Dist
Train = Result.inds_train
Test = Result.inds_test


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
