import numpy as np
import torch
LEN=4000
feats=np.load('features_12328.npy')
print(feats.shape)
Dist=np.zeros([LEN,LEN],np.float32)
for i in range(LEN):
   print(i)
   for j in range(LEN):
       dist= np.dot(feats[i],feats[j])/(np.linalg.norm(feats[i])*(np.linalg.norm(feats[j])))
       Dist[i][j]=dist
np.save('Dist2.npy',Dist)
