# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/20 21:05
# @Author   : Perye(Li Pengyu)
# @FileName : indexed_matrix.py
# @Software : PyCharm

import sys
import weakref
import operator
import random
from array import array
from pathlib import Path
from functools import partial
import numpy as np

from pygraphblas.base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check as _base_check,
    _error_codes,
    _build_range,
    _get_select_op,
    _get_bin_op,
    GxB_INDEX_MAX,
    GraphBLASException,
)

from pygraphblas import Matrix, BOOL, types
from bisect import bisect_left

from pygraphblas.binaryop import current_binop
from pygraphblas.monoid import Monoid
from pygraphblas.semiring import Semiring


def _check(obj, res):
    if res != lib.GrB_SUCCESS:
        error_string = ffi.new("char**")
        error_res = lib.GrB_Matrix_error(error_string, obj._matrix[0])
        if error_res != lib.GrB_SUCCESS:  # pragma: nocover
            raise GraphBLASException(
                "Cannot get error, GrB_Matrix_error itself returned an error."
            )
        raise _error_codes[res](ffi.string(error_string[0]))


class IndexedMatrix(Matrix):
    # row_ptr = [0]

    def __init__(self, matrix, typ=None):
        super().__init__(matrix)
        self.row_ptr = [0]

    def __del__(self):
        _check(self, lib.GrB_Matrix_free(self._matrix))

    def update_indices(self):
        current_row = 0
        n = 0
        for i in self.I:
            if i > current_row:
                self.row_ptr.extend([n] * (i - current_row))
                current_row = i
        # for i in range(self.nvals):
        #     if self.I[i] > current_row:
        #         self.row_ptr.extend([i] * (self.I[i] - current_row))
        #         current_row = self.I[i]
        self.row_ptr.append(self.nvals)

    @classmethod
    def from_lists(cls, I, J, V=None, nrows=None, ncols=None, typ=None):
        matrix = super().from_lists(I, J, V=V, nrows=nrows, ncols=ncols, typ=None)
        matrix.update_indices()
        return matrix

    def eadd(
        self,
        other,
        add_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        func = lib.GrB_Matrix_eWiseAdd_BinaryOp
        if add_op is None:
            add_op = current_binop.get(NULL)
        elif isinstance(add_op, Monoid):
            func = lib.GrB_Matrix_eWiseAdd_Monoid
        elif isinstance(add_op, Semiring):
            func = lib.GrB_Matrix_eWiseAdd_Semiring

        mask, accum, desc = self._get_args(mask, accum, desc)

        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Matrix*")
            _check(self, lib.GrB_Matrix_new(_out, typ._gb_type, self.nrows, self.ncols))
            out = IndexedMatrix(_out, typ)

        if add_op is NULL:
            add_op = out.type._default_addop()

        _check(
            self,
            func(
                out._matrix[0],
                mask,
                accum,
                add_op.get_op(),
                self._matrix[0],
                other._matrix[0],
                desc,
            ),
        )
        return out

    def cal_division_idx(self, ninstances, nth, unit='nzz'):
        idx_list = []
        if 'nzz' == unit:
            block_size = self.nvals // ninstances
            if nth == 0:
                idx_list = [(0, 0), (bisect_left(self.row_ptr, block_size) - 1, self.cols[block_size - 1])]
            else:
                idx_list.append((bisect_left(self.row_ptr, nth * block_size) - 1, self.cols[nth * block_size - 1]))
        elif 'row' == unit:
            block_size = self.nrows // ninstances
            idx_list = [(block_size * nth, 0), (block_size * (nth + 1), 0)]
        return idx_list

    def extract_division_unit(self, start, end, unit='nzz'):
        if 'nzz' == unit:
            # m = super().sparse(self.type, )
            # sub_matrix = self.extract_matrix(slice(idx_list[n - 1] + 1, slice(idx_list[n])))
            pass
        elif 'row' == unit:
            return self.extract_matrix(slice(start[0], max(end[0] - 1, start[0])))


if __name__ == '__main__':
    NUM_NODES = 8
    NUM_EDGES = 32
    # row_indices = [0, 0, 1, 1, 2, 2, 2, 3]
    # col_indices = [0, 1, 1, 3, 2, 3, 4, 5]
    # row_indices = [0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7]
    # col_indices = [0, 2, 3, 5, 7, 0, 1, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0, 3, 4, 5, 7, 0, 2, 7, 2, 3, 6, 0, 1, 4, 5, 6, 7]
    row_indices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7]
    col_indices = [0, 2, 3, 5, 7, 0, 1, 2, 6, 2, 3, 4, 5, 6, 7, 0, 3, 4, 5, 7, 0, 2, 7, 2, 3, 6, 0, 1, 2, 3, 4, 5, 6, 7]
    values = [True for i in range(len(row_indices))]
    # graph = grb.Matrix.sparse(grb.types.BOOL, NUM_NODES, NUM_NODES)
    graph = IndexedMatrix.from_lists(row_indices, col_indices, values, NUM_NODES, NUM_NODES, BOOL)
    graph[0, 0] = False
    graph.clear()
    print(graph)
    print(graph.row_ptr)
    print(graph.cal_division_idx(4, 'nzz'))

    # print(graph[0])
    # print(graph.format)
    # print(graph.S)