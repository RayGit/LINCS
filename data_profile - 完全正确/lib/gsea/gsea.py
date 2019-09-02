import time
import numpy as np


def enrich(up, down, rank, show=False):
    N = len(rank)
    Nr = Nh = len(up)
    # initial the array with the assume that all gene missed
    pos, neg = 1.0 / Nr, - 1.0 / (N - Nh)
    es_up = np.zeros(N) + neg
    es_down = es_up.copy()
    # fix the value where actually hitted
    up_inds, down_inds = rank[up], rank[down]
    es_up[up_inds] = pos
    es_down[down_inds] = pos
    # calculate the sum
    es_down = np.cumsum(es_down)
    es_up = np.cumsum(es_up)
    # just return the finally score
    es = (np.max(np.abs(es_down)) + np.max(np.abs(es_up))) / 2
    return es

def rank(GeneExp, length):
    # Sort means the Indices of elements in GeneExp according sorting
    # Rank will convert GeneExp‘ value to Rank elmentwisely
    Sort = np.argsort(GeneExp)
    Rank = np.argsort(Sort)
    Up, Down = Sort[:, :length], Sort[:, -length:]
    return Rank, Up, Down

def gsea(GeneExp, length=None, distance=False, verbose=False):
    start = time.time()
    num, nfeat = GeneExp.shape
    if length == None: length = nfeat//20
    print(f'[GSEA processing...] samples: {num}, features: {nfeat}, set_length: {length}')
    ES = np.zeros((num, num))
    Rank, Up, Down = rank(GeneExp, length)
    if verbose: total = ((num-1)*num)//2; unit = total/10
    # For faster speed!
    for i, rank_i in enumerate(Rank[:-1]):
        up_i, down_i = Up[i], Down[i]
        for j, rank_j in enumerate(Rank[i+1:], i+1):
            up_j, down_j = Up[j], Down[j]
            ES[i,j] = enrich(up_j, down_j, rank_i)
            # titto
            ES[j,i] = enrich(up_i, down_i, rank_j)
            if verbose:
                fnh = (i*(2*num-i-1))//2 + j-i
                if fnh%unit == 0:
                    pct = fnh/float(total) * 100
                    print(f'[GSEA processing...] {pct:2.2f}% finished, {fnh}/{total}')
    print(f'Executing time is {time.time()-start:.4f}s')
    # add self-similarity, average s(i, j) and s(j, i)
    ES = (ES + ES.T) / 2 + np.eye(num)
    if distance: ES = 1 - ES
    return ES