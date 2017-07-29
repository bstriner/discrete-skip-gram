from sklearn.cluster import KMeans


def cluster_km(z, z_k, n_init=1, max_iter=1000):
    km = KMeans(n_clusters=z_k, n_init=n_init,
                max_iter=max_iter)
    enc = km.fit_predict(z)
    return enc
