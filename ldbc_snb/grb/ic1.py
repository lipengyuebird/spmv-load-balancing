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

from mpi4py import MPI
from pygraphblas import *

from ldbc_snb.loader import loader

logging.basicConfig(
    level=logging.DEBUG,
    filename=f'/home/perye/spmv-load-balancing/ldbc_snb/result/{sys.argv[0].split("/")[-2]}-{os.path.basename(sys.argv[0]).replace(".py", "")}.log',
    filemode='a',
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
)

options_set(
    nthreads=1
)


logging.info('%s' % 'start')

person_knows_person, pkp_node = loader.load_matrix_from_file('/home/perye/spmv-load-balancing/ldbc_snb/sf100_dataset/Person_knows_Person.json')
logging.info('%s' % 'data_loaded')

g = person_knows_person
g.update_indices()

mask_list = [1 for i in range(pkp_node)]

mask0 = Vector.from_list(mask_list)
logging.info('%s' % 'spmv_start')
mask1 = g.mxv(mask0)
mask2 = g.mxv(mask1)
mask3 = g.mxv(mask2)
logging.info('%s' % 'spmv_done')

res = mask1 + mask2 + mask3

