import warnings

import scipy.sparse 
import numpy as np

from solver.util.math import mean_center_col

class CovConstructor:
    def __init__(self, data, nbatch=None):
        '''
        Important: self.data is ALWAYS centered
        '''
        self.data = mean_center_col(data)
        self.ncol = self.data.shape[1]
        self.nrow = self.data.shape[0]
        self.nbatch = nbatch
        self._set_batch_anchors()
    def _set_batch_anchors(self):
        ncol = self.ncol
        batch_size = ncol // self.nbatch
        if batch_size * self.nbatch < ncol:
            batch_size += 1
        if batch_size < 10 and self.nbatch > 5:
            raise ValueError('Too many batches. Exit.')
        self.batch_anchors = []
        for i in range(self.nbatch):
            start = i * batch_size
            end = min((i + 1) * batch_size, ncol)
            self.batch_anchors.append([start, end])
    def _flatten_2d_mat(self, mat):
        row_index_mat = np.tile(np.arange(mat.shape[0]), reps=(mat.shape[1], 1)).T
        row = row_index_mat.flatten()
        del row_index_mat
        col_index_mat = np.tile(np.arange(mat.shape[1]), reps=(mat.shape[0], 1))
        col = col_index_mat.flatten()
        del col_index_mat
        return row, col, mat.flatten()
    @staticmethod
    def _get_fn(output_prefix, suffix):
        if output_prefix is None:
            fn = None
        else:
            fn = output_prefix + suffix
        return fn
    def compute_cov(self, mode, output_prefix=None, param=None):
        if mode == 'cap':
            suffix = '.cap.npz'
            fn = self._get_fn(output_prefix, suffix)
            mat = self.compute_to_cap_npz(fn, threshold=param)
        elif mode == 'banded':
            suffix = '.banded.npz'
            fn = self._get_fn(output_prefix, suffix)
            mat = self.compute_to_banded_npz(fn, band_size=param)
        return mat
    def _compute_cov(self, s1, e1, s2, e2, flatten=True, triu=True):
        '''
        Given submatrix index: 
            matrix1 = [:, s1 : e1], matrix2 = [:, s2 : e2]
        Return: 
            Pairwise covariance between column in matrix1 and column in matrix2.
            Elements are returned in row, col, val format (flatten = True). 
            And only row <= col ones are returned.
            But if flatten = False, triu could be set to False
            to return the full matrix.
        Formula:
            covariance = col1 * col2 / (self.nrow - 1) (col is centered in __init__) 
        '''
        tmp = np.einsum('ni,nj->ij', self.data[:, s1 : e1], self.data[:, s2 : e2]) / (self.nrow - 1)
        if flatten is False:
            if triu is True:
                return np.triu(tmp)
            else:
                return tmp
        row, col, val = self._flatten_2d_mat(tmp)
        row += s1
        col += s2
        to_keep = row <= col
        row, col, val = row[to_keep], col[to_keep], val[to_keep]
        del tmp
        return row, col, val
    def compute_to_banded_npz(self, fn, band_size=100):
        if not isinstance(band_size, int) or band_size < 0:
            raise ValueError('Band_size is integer in compute_to_cap_npz and needs to be non-negative.')
        row_all, col_all, value_all = [], [], []
        for i, (s1, e1) in enumerate(self.batch_anchors):
            for j, (s2, e2) in enumerate(self.batch_anchors):
                if i > j:
                    continue
                if s2 > e1 + band_size - 1:
                    continue
                row, col, value = self._compute_cov(s1, e1, s2, e2)
                to_keep = col - row <= band_size 
                row, col, value = row[to_keep], col[to_keep], value[to_keep]
                row_all.append(row)
                col_all.append(col)
                value_all.append(value)
        row_all = np.concatenate(row_all, axis=0)
        col_all = np.concatenate(col_all, axis=0)
        value_all = np.concatenate(value_all, axis=0)
        cov_coo = scipy.sparse.coo_matrix(
            (value_all, (row_all, col_all)), 
            shape=(self.ncol, self.ncol)
        )
        if fn is not None:
            save_npz(fn, cov_coo)
        return cov_coo  
    def compute_to_cap_npz(self, fn, threshold=1e-5):
        if not isinstance(threshold, float) or threshold < 0:
            raise ValueError('Threshold is float in compute_to_cap_npz and needs to be non-negative.')
        row_all, col_all, value_all = [], [], []
        for i, (s1, e1) in enumerate(self.batch_anchors):
            for j, (s2, e2) in enumerate(self.batch_anchors):
                if i > j:
                    continue
                row, col, value = self._compute_cov(s1, e1, s2, e2)
                to_keep = np.abs(value) > threshold
                row, col, value = row[to_keep], col[to_keep], value[to_keep]
                row_all.append(row)
                col_all.append(col)
                value_all.append(value)
        row_all = np.concatenate(row_all, axis=0)
        col_all = np.concatenate(col_all, axis=0)
        value_all = np.concatenate(value_all, axis=0)
        cov_coo = scipy.sparse.coo_matrix(
            (value_all, (row_all, col_all)), 
            shape=(self.ncol, self.ncol)
        )
        if fn is not None:
            save_npz(fn, cov_coo)
        return cov_coo    

class SparseCov:
    '''
    Keep positive semi-definite sparse matrix (squared matrix).
    Only keep the upper triangular entries.
    Save as COO matrix in scipy.sparse.
    '''
    def __init__(self, filename=None, matrix=None, dense=False):
        if filename is None and matrix is None:
            raise ValueError('Need either filename or matrix to be set.')
        if filename is not None:
            if dense is True:
                raise ValueError('We don\'t support dense=True when using filename.')
            self._init_with_from_file(filename, dense=dense)
        elif matrix is not None:
            self._init_with_mat(matrix, dense=dense)
        self._check_dim()
        self._set_csr_type()
        self._check_upper_triangular()
    def _set_csr_type(self):
        if isinstance(self.csr, scipy.sparse.csr.csr_matrix):
            self.csr_type = 'scipy_sparse_csr'
        else:
            self.csr_type = 'np_array'
    def _init_with_from_file(self, filename, dense=False):
        self.csr = scipy.sparse.load_npz(filename).tocsr()
    def _init_with_mat(self, mat, dense=False):
        if dense is True:
            if not isinstance(mat, np.ndarray):
                raise TypeError('When dense=True, matrix should be dense.')
            else:
                self.csr = mat
            return 
        if isinstance(mat, scipy.sparse.csr.csr_matrix):
            self.csr = mat
        else:
            self.csr = scipy.sparse.csr_matrix(mat)
    def _check_dim(self):
        if self.csr.shape[0] != self.csr.shape[1]:
            raise ValueError('Wrong dim: dim1 != dim2.')
        self.size = self.csr.shape[0]
    def _check_upper_triangular(self):
        if self.csr_type == 'scipy_sparse_csr':
            num_non_zeros_tril = scipy.sparse.tril(self.csr, k=-1).nnz
            if num_non_zeros_tril > 0:
                warnings.warn('This matrix contains non-zero values in the lower triangular portion (k = -1). These values will be ignored.')
                self.csr = scipy.sparse.triu(self.csr).tocsr()
    def get_diag_as_vec(self):
        return self.csr.diagonal()
    def get_jth_diag(self, j):
        return self.csr[j, j]
    def mul_vec(self, vec):
        if self.csr_type == 'scipy_sparse_csr':
            diag_csr = self.csr.diagonal()
            return self.csr.dot(vec) + self.csr.transpose().dot(vec) - diag_csr * vec
        else:
            return self.csr @ vec
    def get_row_as_vec(self, idx):
        if self.csr_type == 'scipy_sparse_csr':
            colwise_idx = self.csr[:, idx].toarray()[:, 0]
            colwise_idx[idx] = 0
            rowwise_idx = self.csr[idx, :].toarray()[0]
            return colwise_idx + rowwise_idx
        else:
            return self.csr[idx, :]
    def add(self, b, coef1=1, coef2=1):
        '''
        Add another SparseCov b:
        coef1 * self.csr + coef2 * b.csr
        '''    
        if self.csr_type == 'scipy_sparse_csr' and b.csr_type == 'scipy_sparse_csr':
            return SparseCov(matrix=self.csr * coef1 + b.csr * coef2)
        elif self.csr_type == 'np_array' and b.csr_type == 'np_array':
            return SparseCov(matrix=self.csr * coef1 + b.csr * coef2, dense=True)
        else:
            raise TypeError('We need both b and self be scipy_sparse_csr or np_array.')
    def to_mat(self):
        if self.csr_type == 'scipy_sparse_csr':
            return scipy.sparse.triu(self.csr) + scipy.sparse.triu(self.csr, k=1).T 
        else:
            return self.csr 