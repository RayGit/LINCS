import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
LEN=4000
TEST_SIZE=0.5
F_LEN=12328
LEN_TRAIN =int( LEN * (1- TEST_SIZE))
LEN_TEST = int(LEN * TEST_SIZE)
def readTable(name):
    table = pd.read_table(name,sep=',',index_col=0)
    #table = table.sample(frac=1.0)
    table = table.values
    return table
def saveTable(data,name):
    table=pd.DataFrame(data)
    table.to_csv(name,header=None,index=None)

csv1='E:/Master/LINCS/GEO DATA/result/untrt_MCF7_base_2000x12328.csv'
csv2='E:/Master/LINCS/GEO DATA/result/untrt_PC3_base_2000x12328.csv'
trainTable=np.zeros([LEN_TRAIN,12329],np.float32)
validationTable=np.zeros([LEN_TEST,12329],np.float32)
table1= readTable(csv1)
table2= readTable(csv2)
table = np.concatenate([table1,table2],axis=0)
inds = np.arange(LEN)
print(inds)
print(inds.shape)
inds_train, inds_test = train_test_split(inds, test_size=TEST_SIZE)
np.save('inds_train_12328.npy',inds_train)
np.save('inds_test_12328.npy',inds_test)


for i in range(inds_train.shape[0]):
        trainTable[i][:F_LEN]=table[inds_train[i]]
        if(inds_train[i] < 2000 ):
            trainTable[i][F_LEN]=0
        else:
            trainTable[i][F_LEN] = 1
for i in range(inds_test.shape[0]):
        validationTable[i][:F_LEN]=table[inds_test[i]]
        if (inds_test[i] < 2000):
            validationTable[i][F_LEN] = 0
        else:
            validationTable[i][F_LEN] = 1

saveTable(trainTable,'trainsample_12328.csv')
saveTable(validationTable,'validationsample_12328.csv')