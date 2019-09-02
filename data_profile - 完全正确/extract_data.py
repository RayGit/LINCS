import pandas as pd
import gctx as gctx
from pandas import DataFrame
import numpy as np

data_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
gene_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_gene_info.txt"
inst_path = r"E:\GEO DATA\raw\GSE92742_Broad_LINCS_inst_info.txt"

inst_data = pd.read_csv(inst_path, sep='\t', low_memory=False)
gene_data = pd.read_csv(gene_path, sep='\t', low_memory=False)

vehicle_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
vehicle_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
vehicle_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_vehicle')]['inst_id'][:2000]
untrt_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
untrt_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
untrt_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_untrt')]['inst_id'][:2000]
vector_MCF7_inst = inst_data[(inst_data.cell_id == 'MCF7') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]
vector_PC3_inst = inst_data[(inst_data.cell_id == 'PC3') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]
vector_A375_inst = inst_data[(inst_data.cell_id == 'A375') & (inst_data.pert_type == 'ctl_vector')]['inst_id'][:2000]


if __name__ == '__main__':
    # file_path = r"E:\GEO DATA\raw\GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328.gctx"
    # g = GCTX(file_path)
    # print(g.shape)
    # print(g)
    # re = g.read(g.insts[:10],g.genes, frame=True)
    # print(re)
    gene_base = gene_data[gene_data.pr_is_lm == 1]['pr_gene_id']
    gene_all = gene_data['pr_gene_id']
    # print(gene_base)

    g = gctx.GCTX(data_path)
    # v1 = g.read(vehicle_MCF7_inst, gene_base, frame=True)
    # v1.to_csv('E:/GEO DATA/result/vehicle_MCF7_base_2000x978.csv')
    # v2 = g.read(vehicle_PC3_inst, gene_base, frame=True)
    # v2.to_csv('E:/GEO DATA/result/vehicle_PC3_base_2000x978.csv')
    # v3 = g.read(vehicle_A375_inst, gene_base, frame=True)
    # v3.to_csv('E:/GEO DATA/result/vehicle_A375_base_2000x978.csv')
    #
    # u1 = g.read(untrt_MCF7_inst, gene_base, frame=True)
    # u1.to_csv('E:/GEO DATA/result/untrt_MCF7_base_2000x978.csv')
    # u2 = g.read(untrt_PC3_inst, gene_base, frame=True)
    # u2.to_csv('E:/GEO DATA/result/untrt_PC3_base_2000x978.csv')
    # u1['Label'] = 0
    # u1['Tag'] = 'MCF7'
    # u1.drop(['Lable'],axis=1,inplace=True)
    # u2['Label'] = 1
    # u2['Tag'] = 'PC3'
    # u2.drop(['Lable'], axis=1, inplace=True)
    # u3 = pd.concat([u1, u2])
    # Data=u3
    # u3.to_csv('E:/GEO DATA/result/test.csv')
    # u3 = g.read(untrt_A375_inst, gene_base, frame=True)
    # u3.to_csv('E:/GEO DATA/result/untrt_A375_base_2000x978.csv')
    # ve1 = g.read(vector_MCF7_inst, gene_base, frame=True)
    # # ve1.to_csv('E:/GEO DATA/result/vector_MCF7_base_2000x978.csv')
    # ve2 = g.read(untrt_PC3_inst, gene_base, frame=True)
    # # ve2.to_csv('E:/GEO DATA/result/vector_PC3_base_2000x978.csv')
    # ve3 = g.read(untrt_A375_inst, gene_base, frame=True)
    # # ve3.to_csv('E:/GEO DATA/result/vector_A375_base_2000x978.csv')
    #
    # r1 = g.read(vehicle_MCF7_inst, gene_all, frame=True)
    # r1.to_csv('E:/GEO DATA/result/vehicle_MCF7_base_2000x12328.csv')
    # r2 = g.read(vehicle_PC3_inst, gene_all, frame=True)
    # r2.to_csv('E:/GEO DATA/result/vehicle_PC3_base_2000x12328.csv')
    # r3 = g.read(untrt_MCF7_inst, gene_all, frame=True)
    # r3.to_csv('E:/GEO DATA/result/untrt_MCF7_base_2000x12328.csv')
    # r4 = g.read(untrt_PC3_inst, gene_all, frame=True)
    # r4.to_csv('E:/GEO DATA/result/untrt_PC3_base_2000x12328.csv')