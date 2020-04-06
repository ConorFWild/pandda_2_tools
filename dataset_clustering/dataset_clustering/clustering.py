import numpy as np

import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def embed_xmaps(numpy_maps):
    #     # Convert to numpy
    #     numpy_maps = [xmap.get_map_data(sparse=False).as_numpy_array()
    #                   for dtag, xmap
    #                   in xmaps.items()]

    # Convert to sample by feature
    sample_by_feature = np.vstack([np_map.flatten()
                                   for np_map
                                   in numpy_maps])

    # Reduce dimension by PCA
    pca = PCA(n_components=min(50, len(numpy_maps)))
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    # Reduce dimension by TSNE
    tsne = TSNE(n_components=2,
                method="exact")
    sample_by_feature_tsne = tsne.fit_transform(sample_by_feature_pca)

    return sample_by_feature_tsne


def cluster_embedding(xmap_embedding):
    # Perform initial clustering
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                prediction_data=True,
                                min_cluster_size=5,
                                )
    # labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    clusterer.fit(xmap_embedding.astype(np.float64))

    return clusterer


