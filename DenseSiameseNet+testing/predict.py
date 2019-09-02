import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,accuracy_score
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from Densenet import DenseNet
import pandas as pd
LEN=4000
GLEN=12328
def readTable(name):
    table = pd.read_table(name,sep=',',index_col=0)
    #table = table.sample(frac=1.0)
    table = table.values
    return table

csv1='E:/Master/LINCS/GEO DATA/result/untrt_MCF7_base_2000x12328.csv'
csv2='E:/Master/LINCS/GEO DATA/result/untrt_PC3_base_2000x12328.csv'
table1= readTable(csv1)
table2= readTable(csv2)
table = np.concatenate([table1,table2],axis=0)
model=DenseNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.to(device)
model.load_state_dict(torch.load("checkpoints/exp1/model_21.pkl")['weight'])

Dist=np.zeros([LEN,50],np.float32)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
for i in range(LEN):
    img1= np.zeros([12544],np.float32)
    img1[:GLEN]=table [i][:GLEN]
    img1=img1.reshape([1,1,112,112])
    img1 = torch.Tensor(img1).cuda()
    out1,_ = model(img1)
    out1=out1.cpu().data.numpy()
    print(out1.shape)
    Dist[i] = out1[0]

    '''for j in range(LEN):
        print('j:',j)
        img2 = np.zeros([1024], np.float32)
        img2[:GLEN] = table[j][:GLEN]
        img2 = img2.reshape([1,1, 32, 32])
        img2 = torch.Tensor(img2)
        img2 = img2.cuda()
        out2, _ = model(img2)
        dist_cos=cos(out1,out2).cpu().data.numpy()
        Dist[i,j]=dist_cos[0]'''


np.save('features_12328.npy',Dist)











