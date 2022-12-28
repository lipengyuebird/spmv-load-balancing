# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/24 8:53
# @Author   : Perye(Li Pengyu)
# @FileName : get_split_idx.py
# @Software : PyCharm

import os
import sys
sys.path.append('/root/spmv-load-balancing')

import logging

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pygraphblas import *

from util.serialization import to_np_vector
from ldbc_snb.loader import loader

logging.basicConfig(
    level=logging.DEBUG,
    filename=f'/home/perye/spmv-load-balancing/ldbc_snb/result/{sys.argv[0].split("/")[-2]}-{os.path.basename(sys.argv[0]).replace(".py", "")}.log',
    filemode='a',
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

person_knows_person, pkp_node = loader.load_matrix_from_file('/home/perye/spmv-load-balancing/ldbc_snb/sf100_dataset/Person_knows_Person.json')


g = person_knows_person
g.update_indices()
print(pkp_node)
print(g.nrows)
# print(len(g.row_ptr))
#
# size = 4
#
# for unit in ['row', 'nzz']:
#     for rank in range(size):
#         print(g.cal_division_idx(size, rank, unit))
