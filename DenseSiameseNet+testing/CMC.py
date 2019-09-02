import numpy as np
import matplotlib as plt
test_cmc = []  #保存accuracy，记录rank1到rank48的准确率
sort_index = np.argsort(-predict_label, axis=1) #predict_label为模型预测得到的匹配分数矩阵；降序排序，返回匹配分数值从大到小的索引值
actual_index = np.argmax(test_y,1) #test_y为测试样本的真实标签矩阵；返回一列真实标签相对应的最大值的索引值
predict_index = np.argmax(predict_label,1)#返回一列预测标签相对应的最大值的索引值
temp = np.cast['float32'](np.equal(actual_index,predict_index)) #一列相似值，1代表相同，0代表不同
test_cmc.append(np.mean(temp))#rank1
#rank2到rank48
for i in range(sort_index.shape[1]-1):
    for j in range(len(temp)):
        if temp[j]==0:
            predict_index[j] = sort_index[j][i+1]
    temp = np.cast['float32'](np.equal(actual_index,predict_index))
    test_cmc.append(np.mean(temp))
#创建绘图对象
plt.figure()
x = np.arange(0,sort_index.shape[1])
plt.plot(x,test_cmc,color="red",linewidth=2)
plt.xlabel("Rank")
plt.ylabel("Matching Rate")
plt.legend()
plt.title("CMC Curve")
