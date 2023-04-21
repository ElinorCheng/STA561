## higham
def P_s(a, wh):
    # helper function for higham, the second projection
    a = wh.dot(a).dot(wh)
    vals, vecs = np.linalg.eigh(a)
    vals[np.where(vals < 0)] = 0
    a = vecs.dot(np.diag(vals)).dot(vecs.T)

    wh_inv = np.diag(1 / np.diagonal(wh))

    a = wh_inv.dot(a).dot(wh_inv)
    return a


def P_u(a):
    # helper function for higham, the first projection
    np.fill_diagonal(a, 1)
    return a


def higham_psd(a, tol=None, max_iter=100, weights=None):
    # higham nearest psd
    if tol is None:
        tol = np.spacing(1) * len(a)
    if weights is None:
        weights = np.ones(len(a))
    w_h = np.diag(np.sqrt(weights))
    Y = np.copy(a)
    ds = np.zeros(np.shape(a))
    for i in range(0, max_iter):
        norm_Y_pre = fro_norm(Y)
        R = Y - ds
        X = P_s(R, w_h)

        ds = X - R
        Y = P_u(X)
        norm_Y = fro_norm(Y)

        if -1 * tol < norm_Y - norm_Y_pre < tol:
            break
    return Y

def fro_norm(a):
    return np.sum(np.square(a))
