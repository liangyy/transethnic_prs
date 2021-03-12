from solver.util.sparse_mat import mul_vec

def load_genofile_as_XtX(filename, y):
    if isinstance(filename, str):
        pass
    else:
        # for testing, we skip this loading function
        # filename is a csr matrix
        mat = filename
        return mat.T @ mat, mul_vec(mat, y)