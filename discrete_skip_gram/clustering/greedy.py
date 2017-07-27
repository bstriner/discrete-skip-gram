import numpy as np


def greedy_cluster(losses):
    used = []
    clusters = []
    best_val = None
    while not best_val:
        best_val = None
        best = None
        for k1, v in losses.iteritems():
            if k1 not in used:
                for k2, loss in v.iteritems():
                    if k2 not in used:
                        if (not best_val) or (loss < best_val):
                            best_val = loss
                            best = [k1, k2]
        if best_val:
            used.append(best[0])
            used.append(best[1])
            clusters.append(best)
    return clusters


def recursive_cluster(covariance, depth, eps=1e-8):
    covariance = covariance.astype(np.float32)
    covariance = covariance / np.sum(covariance, axis=None)
    x_k = covariance.shape[1]
    p0 = covariance
    w0 = list(range(x_k))
    while len(w0)>1:
        n = p0.shape[0]
        losses = {}
        for a in range(n):
            losses[a]={}
            for b in range(a+1, n):
                p = p0[a,:] + p0[b,:]
                m = np.sum(p)
                c = p/c
                nll = c*-np.log(eps+c)
                loss = nll*m
                losses[a][b]=loss
        clusters = greedy_cluster(losses)
        p1 = np.zeros((n//2), x_k)
        w1 = []
        for row, clust in enumerate(clusters):
            p1[row,:] = (p0[clust[0],:]) + (p0[clust[1],:])
            w1 = [w0[clust[0]], w0[clust[1]]]
        p0=p1
        w0=w1
    return w0

