'''
Some utils for Computing the performance of Results
'''


import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_curve, roc_auc_score

def knn(Distance, Label, Train, k=1):
    class_num = len(np.unique(Label))
    Predict = np.zeros(len(Label), dtype=np.int)
    for i,each in enumerate(Distance):
        dist = each[Train]
        sort_ind = np.argsort(dist)[:k]
        corr_label = np.array([Label[Train[ind]] for ind in sort_ind])
        if np.shape(np.shape(corr_label))[0]!=1:
            corr_label=np.squeeze(corr_label)
        Predict[i] = np.argmax(np.bincount(corr_label))

    return Predict

def spectral(Distance, Label):
    num = len(Label)
    SC = SpectralClustering(n_clusters=2, affinity='precomputed')
    Pred = SC.fit_predict(Distance)
    if type(Label) is dict:
        Label = np.array([Label[i][1] for i in range(num)])
    Err1 = np.sum(Label ^ Pred) / float(num)
    opposite_Label = np.array([i ^ 1 for i in Label])
    Err2 = np.sum(opposite_Label ^ Pred) / float(num)
    return Err1, Err2

def accuracy(TrueLabel, PredLabel):
    length = len(TrueLabel)
    acc = 0.0
    for i in range(length):
        if TrueLabel[i] == PredLabel[i]:
            acc += 1
    acc = acc / length
    return acc

def pairs_label(Data):
    pairsLabel = []
    if len(Data.shape) == 2:
        row, col = Data.shape
        Data = Data/np.max(Data)
        assert row == col, "Data Matrix must be square!"
        for i in range(row-1):
            for j in range(i+1, col):
                pairsLabel.append(Data[i,j])
    else:
        len_ = len(Data)
        Data=Data.reset_index(drop=True)
        for i in range(len_-1):
            for j in range(i+1, len_):
                if Data[i] == Data[j]:
                    pairsLabel.append(0)
                else:
                    pairsLabel.append(1)
    return pairsLabel

def roc(Distance, Label, part=None, save_figure=None):
    if part:
        Distance = Distance[part, part]
        Label = Label[part]

    trueLabel = pairs_label(Label)
    if type(Distance) is dict:
        predLabel = {}
        for method, Dist in Distance.items():
            predLabel[method] = pairs_label(Dist)

    auc, fpr, tpr = {}, {}, {}
    for method,score in predLabel.items():
        auc[method] = roc_auc_score(trueLabel, score)
        fpr[method], tpr[method], _ = roc_curve(trueLabel, score, pos_label=1)

    if save_figure:
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        plt.figure("ROC")
        plt.ylim(0,1.01)
        plt.plot([0, 1], [0, 1], 'k--')

        for method in predLabel.keys():
            label_str = method + "(%.3f)" % auc[method]
            plt.plot(fpr[method], tpr[method], label=label_str)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    return auc, fpr, tpr
