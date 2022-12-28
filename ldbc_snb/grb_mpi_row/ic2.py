# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/14 19:24
# @Author   : Perye(Li Pengyu)
# @FileName : ic2.py
# @Software : PyCharm
import os
import sys
sys.path.append('/root/spmv-load-balancing')

import logging

import numpy as np
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

comm = MPI.COMM_WORLD

logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'starting'))

person_knows_person, pkp_node = loader.load_matrix_from_file('/home/perye/spmv-load-balancing/ldbc_snb/sf100_dataset/Person_knows_Person.json')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'data_loaded'))

g = person_knows_person
g.update_indices()
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'row_ptr_calculated'))

mask_list = [1 for i in range(pkp_node)]
mask = Vector.from_list(mask_list)
res = np.zeros(pkp_node, np.int8)
division_indices = g.cal_division_idx(comm.size, comm.rank, 'row')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'div_indices_calculated'))

my_m = g.extract_division_unit(*division_indices, 'row')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'div_unit_extracted'))

my_new_vec = my_m.mxv(mask)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'spmv_done'))

my_new_vec = to_np_vector(my_new_vec, division_indices[0][0], pkp_node)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'serialized'))

comm.Allreduce([my_new_vec, MPI.INT], [res, MPI.INT], MPI.SUM)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'done'))
