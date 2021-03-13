def list_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for i, j in zip(l1, l2):
        if i != j:
            return False
    return True

def merge_two_lists(l1, l2):
    out = []
    for i, j in zip(l1, l2):
        out.append(f'{i}_{j}')
    return out
