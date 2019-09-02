import os
import h5py
import numpy as np
import pandas as pd

class GCTX(object):
    def __init__(self, root=None, verbose=True):
        self.root = root
        self.file = os.path.basename(root)
        self.verbose = verbose
        self._open_gctx()

    def __repr__(self):
        return f'GCTX (file = {self.file})'

    def __str__(self):
        return f'File: {self.file},\nMatrix: numpy.ndarray of size {self.matrix.shape}'

    def _open_gctx(self):
        if self.verbose:
            print(f'H5py File: {self.file} opened!')
        self.gctx_data = h5py.File(self.root, 'r')
        self.matrix = self.gctx_data['0/DATA/0/matrix']
        self.insts = get_col(self.gctx_data)
        self.genes = get_row(self.gctx_data)
        self._shape = self.matrix.shape

    def _close_gctx(self):
        self.gctx_data.close()

    @property
    def shape(self):
        return self._shape

    def read(self, insts=None, genes=None, insts_inds=None, genes_inds=None, convert_to_double=False, frame=False):
        assert (insts is not None or insts_inds is not None),\
        'Neither instances list nor indices is provided'
        if insts_inds is None:
            insts_inds = get_sub_inds(self.insts, insts)
        else:
            insts = self.insts[insts_inds]

        if (genes_inds is None) & (genes is None):#为什么和inst_inds不一样呢？
            genes = self.genes
            genes_inds = range(len(genes))
            sub_gene = False
        else:
            sub_gene = True
            if genes_inds is None:
                genes_inds = get_sub_inds(self.genes, genes)
            else:
                genes = self.genes[genes_inds]

        if self.verbose:
            print(f'GCTX_READER: reading ({len(insts)}, {len(genes)}) shaped matrix data...')

        # Indexing elements must be in increasing order
        insts_pairs = sorted(zip(insts, insts_inds), key=lambda x: x[1])
        genes_pairs = sorted(zip(genes, genes_inds), key=lambda x: x[1])
        insts, insts_inds = zip(*insts_pairs)
        genes, genes_inds = zip(*genes_pairs)

        result = self.matrix[insts_inds, :]
        if sub_gene:
            result = result[:, genes_inds]
        if frame:
            result = pd.DataFrame(result, index=insts, columns=genes)
        return result

def get_row(h5file):
    return list(map(int, h5file['0/META/ROW/id'][:]))

def get_col(h5file):
    return list(map(bytes.decode, h5file['0/META/COL/id'][:]))

def get_sub_inds(full_list, match_list):
    missings = set(match_list) - set(full_list)
    if missings:
        raise Exception(f'The following items in the match list did not have matching cids: {missings}')
    full_idx = dict(zip(full_list, range(len(full_list))))
    matches  = [full_idx[x] for x in match_list]
    return matches


