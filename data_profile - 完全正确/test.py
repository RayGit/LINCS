import h5py
import numpy as np
import pandas as pd

import lib.gsea.gsea as gsea

level3_file = r'E:\GEODATA\raw\GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328.gctx'

level3 = h5py.File(level3_file, 'r')

data = level3['0/DATA/0/matrix'][:100, :]

es = gsea.gsea(data, length=250, verbose=True)

level3.close()