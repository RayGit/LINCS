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
        print(type(insts))
        print("s$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
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


# if __name__ == '__main__':
#     # file_path = r"E:\GEO DATA\raw\GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328.gctx"
#     # g = GCTX(file_path)
#     # print(g.shape)
#     # print(g)
#     # re = g.read(g.insts[:10],g.genes, frame=True)
#     # print(re)
#     data_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
#     gene_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_gene_info.txt"
#     inst_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_inst_info.txt"

#     inst_data = pd.read_csv(inst_path, sep='\t', low_memory=False)
#     gene_data = pd.read_csv(gene_path, sep='\t', low_memory=False)

#     vehicle_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
#     vehicle_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
#     vehicle_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
#     untrt_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
#     untrt_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
#     untrt_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
#     vector_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]
#     vector_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]
#     vector_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]

#     gene_base = gene_data[gene_data.pr_is_lm == 1]['pr_gene_id']
#     gene_all=gene_data['pr_gene_id']
#     # print(gene_base)

#     g = GCTX(data_path)
#     v1 = g.read(vehicle_MCF7_inst, gene_base, frame=True)
#     v1.to_csv('E:/GEO DATA/result/vehicle_MCF7_base_2000x978.csv', float_format='%.2f')
#     v2 = g.read(vehicle_PC3_inst, gene_base, frame=True)
#     v2.to_csv('E:/GEO DATA/result/vehicle_PC3_base_2000x978.csv', float_format='%.2f')
#     v3 = g.read(vehicle_A375_inst, gene_base, frame=True)
#     v3.to_csv('E:/GEO DATA/result/vehicle_A375_base_2000x978.csv', float_format='%.2f')

#     u1 = g.read(untrt_MCF7_inst, gene_base, frame=True)
#     u1.to_csv('E:/GEO DATA/result/untrt_MCF7_base_2000x978.csv', float_format='%.2f')
#     u2 = g.read(untrt_PC3_inst, gene_base, frame=True)
#     u2.to_csv('E:/GEO DATA/result/untrt_PC3_base_2000x978.csv', float_format='%.2f')
#     u3 = g.read(untrt_A375_inst, gene_base, frame=True)
#     u3.to_csv('E:/GEO DATA/result/untrt_A375_base_2000x978.csv', float_format='%.2f')

#     ve1 = g.read(vector_MCF7_inst, gene_base, frame=True)
#     ve1.to_csv('E:/GEO DATA/result/vector_MCF7_base_2000x978.csv', float_format='%.2f')
#     ve2 = g.read(untrt_PC3_inst, gene_base, frame=True)
#     ve2.to_csv('E:/GEO DATA/result/vector_PC3_base_2000x978.csv', float_format='%.2f')
#     ve3 = g.read(untrt_A375_inst, gene_base, frame=True)
#     ve3.to_csv('E:/GEO DATA/result/vector_A375_base_2000x978.csv', float_format='%.2f')

#     r1 = g.read(vehicle_MCF7_inst, gene_all, frame=True)
#     r1.to_csv('E:/GEO DATA/result/vehicle_MCF7_base_2000x12328.csv', float_format='%.2f')
#     r2 = g.read(vehicle_PC3_inst, gene_all, frame=True)
#     r2.to_csv('E:/GEO DATA/result/vehicle_PC3_base_2000x12328.csv', float_format='%.2f')
#     r3 = g.read(untrt_MCF7_inst, gene_all, frame=True)
#     r3.to_csv('E:/GEO DATA/result/untrt_MCF7_base_2000x12328.csv', float_format='%.2f')
#     r4 = g.read(untrt_PC3_inst, gene_all, frame=True)
#     r4.to_csv('E:/GEO DATA/result/untrt_PC3_base_2000x12328.csv', float_format='%.2f')