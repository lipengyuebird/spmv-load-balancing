# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/23 19:24
# @Author   : Perye(Li Pengyu)
# @FileName : ic2.py
# @Software : PyCharm

import sys
import time

sys.path.append('/home/perye/spmv-load-balancing')

import random

import numpy as np
from mpi4py import MPI

from util.serialization import to_np_vector
from pygraphblas import *

from ldbc_snb.loader import loader
person_knows_person, pkp_node = loader.load_matrix_from_file('/home/perye/spmv-load-balancing/ldbc_snb/sf10_dataset/Person_knows_Person1.json')

g = person_knows_person
g.update_indices()

mask_list = [1 for i in range(pkp_node)]
# mask_list = [0 for i in range(pkp_node)]
# mask_list[5] = 1

print(mask_list)
mask = Vector.from_list(mask_list)
res = np.zeros(pkp_node, np.int8)
comm = MPI.COMM_WORLD

# comm.Barrier()
# t_start = MPI.Wtime()

division_indices = g.cal_division_idx(comm.size, comm.rank, 'row')
t_start = time.time()
my_new_vec = g.extract_division_unit(*division_indices, 'row').mxv(mask)
t_diff = time.time() - t_start

my_new_vec = to_np_vector(my_new_vec, division_indices[0][0], pkp_node)

comm.Allreduce(
    [my_new_vec, MPI.INT],
    [res, MPI.INT],
    MPI.SUM,
)

# comm.Barrier()
# t_diff = time.time() - t_start

print("finished in %5.2fs: %5.2f hops per second" %
       (t_diff, 1 / t_diff)
       )
print("============================================================================")
