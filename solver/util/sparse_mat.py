def get_diag_as_vec(mat_csr):
    return mat_csr.diagonal()
def mul_vec(mat_csr, vec):
    return mat_csr @ vec
def get_row_as_vec(mat_csr, idx):
    return mat_csr[idx, :].toarray()[0]