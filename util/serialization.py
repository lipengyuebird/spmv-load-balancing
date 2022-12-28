# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/23 18:23
# @Author   : Perye(Li Pengyu)
# @FileName : serialization.py
# @Software : PyCharm

from pygraphblas import Vector
import numpy as np


def to_np_vector(v: Vector, offset=0, size=None, fill_nan_with=0):
    if not size:
        size = v.size
    if fill_nan_with is None:
        return v.npV
    else:
        npv = np.zeros(size, np.int8)
        print(offset)
        print(v)
        print(npv)
        val_it = iter(v.V)
        for idx in v.I:
            npv[offset + idx] = next(val_it)
        print(npv)
        return npv