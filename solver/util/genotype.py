def load_genofile_as_XtX(filename):
    if isinstance(filename, str):
        pass
    else:
        # for testing, we skip this loading function
        # filename is a csr matrix
        return filename