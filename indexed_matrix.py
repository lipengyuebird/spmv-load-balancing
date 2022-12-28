# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/10 21:05
# @Author   : Perye(Li Pengyu)
# @FileName : indexed_matrix.py
# @Software : PyCharm

import sys
sys.path.append('/root/spmv-load-balancing')


from pygraphblas.base import (
    lib,
    ffi,
    NULL,
    NoValue,
)

from pygraphblas import Matrix, BOOL, types
from pygraphblas.binaryop import current_binop
from pygraphblas.monoid import Monoid
from pygraphblas.semiring import Semiring

from bisect import bisect_left


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
        self.row_ptr = [0]
        current_row = 0
        n = 0
        for i in self.I:
            if i > current_row:
                self.row_ptr.extend([n] * (i - current_row))
                current_row = i
            n += 1
        self.row_ptr.append(self.nvals)

    @classmethod
    def from_lists(cls, I, J, V=None, nrows=None, ncols=None, typ=None):
        matrix = super().from_lists(I, J, V=V, nrows=nrows, ncols=ncols, typ=None)
        matrix.update_indices()
        return matrix

    def eadd(self, other, add_op=None, cast=None, out=None, mask=None, accum=None, desc=None,):
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

        _check(self, func(out._matrix[0], mask,accum, add_op.get_op(), self._matrix[0], other._matrix[0], desc))
        return out

    def cal_division_idx(self, ninstances, nth, unit='nzz'):
        idx_list = []
        if 'nzz' == unit:
            block_size = self.nvals // ninstances
            if nth == 0:
                idx_list = [(0, 0), (bisect_left(self.row_ptr, block_size) - 1, self.cols[block_size - 1])]
            else:
                idx_list = [
                    (bisect_left(self.row_ptr, nth * block_size) - 1, self.cols[nth * block_size - 1]),
                    (bisect_left(self.row_ptr, (nth + 1) * block_size) - 1, self.cols[(nth + 1) * block_size - 1])
                ]
        elif 'row' == unit:
            block_size = self.nrows // ninstances
            idx_list = [(block_size * nth, 0), (block_size * (nth + 1), 0)]
        return idx_list

    def extract_division_unit(self, start, end, unit='nzz'):
        if 'nzz' == unit:
            m = self.extract_matrix(slice(start[0], max(end[0], start[0])))
            for i in range(0, start[1]):
                del(m[0, i])
            for i in range(end[1], m.nrows):
                del(m[end[0] - start[0], i])
            return m
        elif 'row' == unit:
            return self.extract_matrix(slice(start[0], max(end[0] - 1, start[0])), slice(0, self.ncols - 1))


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
    graph.update_indices()
    print(graph.row_ptr)

    # print(graph[0])
    # print(graph.format)
    # print(graph.S)