# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/14 19:24
# @Author   : Perye(Li Pengyu)
# @FileName : ic1.py
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

logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'start'))

person_knows_person, pkp_node = loader.load_matrix_from_file('/home/perye/spmv-load-balancing/ldbc_snb/sf100_dataset/Person_knows_Person.json')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'data_loaded'))

g = person_knows_person
g.update_indices()
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'row_ptr_updated'))

mask_list = [1 for i in range(pkp_node)]
mask0 = Vector.from_list(mask_list)
mask1 = np.zeros(pkp_node, np.int8)
division_indices = g.cal_division_idx(comm.size, comm.rank, 'nzz')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'div_indices_calculated'))

my_m = g.extract_division_unit(*division_indices, 'nzz')
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'div_unit_extracted'))

res1 = my_m.mxv(mask0)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'spmv_done_1'))

res1 = to_np_vector(res1, division_indices[0][0], pkp_node)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'serialized_1'))

comm.Allreduce([res1, MPI.INT], [mask1, MPI.INT], MPI.SUM)
comm.Barrier()  # synchronize
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'hop1_done'))

mask1 = Vector.from_list(mask1.tolist())
mask2 = np.zeros(pkp_node, np.int8)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'hop2_start'))

res2 = my_m.mxv(mask1)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'spmv_done_2'))

res2 = to_np_vector(res2, division_indices[0][0], pkp_node)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'serialized_2'))

comm.Allreduce([res2, MPI.INT], [mask2, MPI.INT], MPI.SUM)
comm.Barrier()
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'hop2_done'))

mask2 = Vector.from_list(mask2.tolist())
mask3 = np.zeros(pkp_node, np.int8)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'hop3_start'))

res3 = my_m.mxv(mask2)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'spmv_done_2'))

res3 = to_np_vector(res3, division_indices[0][0], pkp_node)
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'serialized_3'))

comm.Allreduce([res3, MPI.INT], [mask3, MPI.INT], MPI.SUM)
comm.Barrier()
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'hop3_done'))

mask3 = Vector.from_list(mask3.tolist())
res = mask1 + mask2 + mask3
logging.info('instance %d of %d - %s' % (comm.rank, comm.size, 'all_done'))
