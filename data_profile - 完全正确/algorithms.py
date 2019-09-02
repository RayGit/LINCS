"""
Algorithms Calculated Here including {'EUC', 'GSEA', 'LMNN', 'ITML', 'SDML', 'LSML', 'NCA', 'LFDA', 'RCA'}

"""

import numpy as np
from sklearn.preprocessing import Imputer
from metric_learn import LMNN, NCA, LFDA
from metric_learn import ITML_Supervised, SDML_Supervised, LSML_Supervised, RCA_Supervised

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

import lib.gsea.gsea as gsea

#按照子类索引在母集搜索得出对应的字典
def select(options_all, sub_list):
    options ={}
    for method, option in options_all.items():
        if method in sub_list:
            options[method] = option
    return options

class ALGO(object):
    """docstring for ALGO"""

    def __init__(self, GeneExp, Label,  test_size=0.5, **options):
        self.GeneExp = GeneExp
        self.Label = Label
        # self.Tag = Tag
        self.test_size = test_size
        self.options = options

        self.algo = {'BASE': self.process_base, 'GSEA': self.process_gsea,
                     'LMNN': self.process_lmnn, 'ITML': self.process_itml,
                     'SDML': self.process_sdml, 'LSML': self.process_lsml,
                     'NCA':  self.process_nca,  'LFDA': self.process_lfda,
                     'RCA':  self.process_rca,}

        self.Trans = {}
        self.Dist = {}
        self.split_data(GeneExp, Label)
        self.run_processes()

    #随机产生训练样本和测试样本，比列0.5
    def split_data(self, GeneExp, Label):
        '''Generating training and testing data and label'''
        inds = np.arange(len(Label-1))
        inds_train, inds_test = train_test_split(inds, test_size=self.test_size)
        self.GeneExp_train = GeneExp.iloc[inds_train]
        self.Label_train = Label.iloc[inds_train]
        self.inds_train = inds_train
        self.inds_test = inds_test



    # metric=eculidean
    def process_base(self, **option):
        '''baseline distance metrics using pairwise_distances'''
        GeneExp = self.GeneExp
        # self.Dist['EUC'] = pairwise_distances(GeneExp, metric='cosine')
        self.Dist['EUC'] = pairwise_distances(GeneExp, metric='cosine')

    # length=None, p=1, distance=False, verbose=False
    def process_gsea(self, **option):
        '''calculating enrichment score by GSEA algorithm'''
        GeneExp = self.GeneExp.values
        self.Dist['GSEA'] = gsea.gsea(GeneExp, **option)

    # k=3, min_iter=50, max_iter=1000, learn_rate=1e-7, regularization=0.5, convergence_tol=0.001, verbose=False
    def process_lmnn(self, **option):
        '''Metric Learning algorithm: LMNN'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        lmnn = LMNN(**option)
        lmnn.fit(GeneExp, Label)
        self.Trans['LMNN'] = lmnn.transformer()

    # gamma=1., max_iters=1000
    def process_itml(self, **option):
        '''Metric Learning algorithm: ITML'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        itml = ITML_Supervised(**option)
        itml.fit(GeneExp, Label)
        self.Trans['ITML'] = itml.transformer()

    # balance_param=0.5, sparsity_param=0.01
    def process_sdml(self, **option):
        '''Metric Learning algorithm: SDML'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        sdml = SDML_Supervised(**option)
        sdml.fit(GeneExp, Label)
        self.Trans['SDML'] = sdml.transformer()

    # num_constraints = 2000
    def process_lsml(self, **option):
        '''Metric Learning algorithm: LSML'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        lsml = LSML_Supervised(**option)
        lsml.fit(GeneExp, Label)
        self.Trans['LSML'] = lsml.transformer()

    # max_iter=1000, learning_rate=0.01
    def process_nca(self, **option):
        '''Metric Learning algorithm: NCA'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        nca = NCA(**option)
        nca.fit(GeneExp, Label)
        self.Trans['NCA'] = nca.transformer()

    # k=2, dim=50
    def process_lfda(self, **option):
        '''Metric Learning algorithm: LFDA'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train

        lfda = LFDA(**option)
        lfda.fit(GeneExp, Label)
        self.Trans['LFDA'] = lfda.transformer()

    # num_chunks=30, chunk_size=2
    def process_rca(self, **option):
        '''Metric Learning algorithm: RCA'''
        GeneExp = self.GeneExp_train
        Label = self.Label_train
        rca = RCA_Supervised(num_chunks=30, chunk_size=2)
        rca.fit(GeneExp, Label)
        self.Trans['RCA'] = rca.transformer()

    def run_processes(self):
        options = self.options
        algo = self.algo
        for method,option in options.items():
            print("ALGO: Process %s is going..." % method)
            algo[method](**option)
        for method in self.Trans:
            L = self.Trans[method]
            # self.Dist[method] = pairwise_distances(self.GeneExp.dot(L),metric='cosine')
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
            L= imp.fit_transform(L)
            self.Dist[method] = pairwise_distances(self.GeneExp.dot(L), metric='cosine')
