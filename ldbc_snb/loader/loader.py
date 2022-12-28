# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/14 19:31
# @Author   : Perye(Li Pengyu)
# @FileName : loader.py
# @Software : PyCharm

import sys

sys.path.append('/root/spmv-load-balancing')

import json
from indexed_matrix import IndexedMatrix


def load_matrix_from_file(path):
    with open(path, 'r') as f:
        idx = json.load(f)
    nnode = max(max(idx[0]), max(idx[1])) + 1
    nnode = nnode + 4 - nnode % 4  # buffer size alignment
    return IndexedMatrix.from_lists(idx[0], idx[1], [1 for i in range(len(idx[0]))], nnode, nnode), nnode
