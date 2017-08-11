from sklearn.cluster import KMeans
from ..flat_validation import validate_encoding_flat

def cluster_km(z, z_k, n_init=1, max_iter=1000):
    km = KMeans(n_clusters=z_k, n_init=n_init,
                max_iter=max_iter)
    enc = km.fit_predict(z)
    return enc

def validate_cluster_km(z, z_k, cooccurrence):
    enc = cluster_km(z=z, z_k=z_k)
    nll = validate_encoding_flat(cooccurrence=cooccurrence, enc=enc)
    return nll