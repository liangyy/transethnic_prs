from solver.util.math import l2_norm_sq, l1_norm

def obj_original(beta, x, y, w1, w2):
    resid = y - x @ beta
    return l2_norm_sq(resid) + w1 * l1_norm(beta) + w2 * l2_norm_sq(beta) 

def obj_model1(beta, A, b, w1, w2):
    return beta.T @ A @ beta - 2 * b.T @ beta + w1 * l1_norm(beta) + w2 * l2_norm_sq(beta) 