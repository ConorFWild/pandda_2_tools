from typing import NamedTuple, Dict
import logging

logging.basicConfig(filename='luigi.log', level=logging.INFO)

import argparse
from functools import partial
from pathlib import Path
import os
import shutil
import time

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS

from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.manifold import TSNE

import hdbscan

import luigi

luigi.configuration.get_config().set('logging', 'log_level', 'INFO')

import dataset_clustering
# from dataset_clustering.cluster import cluster_datasets_rsa as cluster_datasets
from dataset_clustering.cluster import cluster_datasets_rsa as cluster_datasets
from dataset_clustering.data_handling import copy_datasets
from dataset_clustering.cluster import cluster_datasets_luigi

from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

import clipper_python

from mdc3.types.datasets import (parse_pandda_input,
                                 parse_pandda_input_for_regex,
                                 )
from mdc3.functions.utils import res_from_mtz_file

from XChemDB.xchem_db import XChemDB

from dataset_clustering.clustering import (embed_xmaps,
                                           cluster_embedding,
                                           )

from dataset_clustering.data_handling import (output_labeled_embedding_csv,
                                              get_map_clusters,
                                              )

from dataset_clustering.luigi_lib import (CopyDataset,
                                          AlignDataset,
                                          ClusterDatasets,
                                          NormaliseStructure,
                                          normalise_structure,
                                          )

from dataset_clustering.xmap_utils import (make_mean_map,
                                           output_mean_map,
                                           output_mean_nxmap,
                                           ccp4_path_to_np,
                                           load_ccp4_map,
                                           )

from mdc3.types.real_space import MCDXMap


class ClusterXCDBArgs(NamedTuple):
    root_path: Path
    out_dir: Path
    n_procs: int
    clean_run: bool
    pdb_regex: str
    mtz_regex: str
    structure_factors: str
    align: bool
    method: str

    # xcdb_path: Path


class ClusterArgs(NamedTuple):
    data_dirs: Path
    out_dir: Path
    processor: str
    n_procs: int


def get_cmd_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-r", "--root_path",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-o", "--out_dir",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-n", "--n_procs",
                        type=str,
                        help="Number of processes to start",
                        required=True
                        )

    parser.add_argument("-c", "--clean_run",
                        type=bool,
                        help="Number of processes to start",
                        default=True,
                        )

    parser.add_argument("--mtz_regex",
                        type=str,
                        help="Number of processes to start",
                        default="dimple.mtz",
                        )

    parser.add_argument("--pdb_regex",
                        type=str,
                        help="Number of processes to start",
                        default="dimple.pdb",
                        )
    parser.add_argument("--structure_factors",
                        type=str,
                        help="Number of processes to start",
                        default="FWT,PHWT",
                        )
    parser.add_argument("--method",
                        type=str,
                        help="Number of processes to start",
                        default="mixture",
                        )

    return parser


def cat_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs,
                   ignore_index=True,
                   )
    return df


def get_args():
    cmd_args = get_cmd_args().parse_args()

    args = ClusterXCDBArgs(root_path=Path(cmd_args.root_path),
                           out_dir=Path(cmd_args.out_dir),
                           n_procs=int(cmd_args.n_procs),
                           clean_run=bool(cmd_args.clean_run),
                           pdb_regex=cmd_args.pdb_regex,
                           mtz_regex=cmd_args.mtz_regex,
                           structure_factors=cmd_args.structure_factors,
                           method=cmd_args.method,
                           align=False,
                           )
    return args


def get_datasets(root_path,
                 mtz_regex,
                 pdb_regex,
                 ):
    datasets = parse_pandda_input(Path(root_path),
                                  mtz_regex=mtz_regex,
                                  pdb_regex=pdb_regex,
                                  )
    return datasets


def truncate_datasets(datasets, res_limit):
    truncated_datasets = {}
    for dtag, d in datasets.items():
        if res_from_mtz_file(d["mtz_path"]).limit > res_limit:
            print("\tRemoving dataset {}: bad resolution".format(dtag))
            continue
        else:
            truncated_datasets[dtag] = d

    return truncated_datasets


def setup_output(out_dir):
    system_out_dir = out_dir
    system_copied_dir = system_out_dir / "copied"
    system_aligned_dir = system_out_dir / "maps"
    system_processed_dir = system_out_dir / "processed"

    if args.clean_run:

        try:
            shutil.rmtree(system_out_dir.resolve(),
                          ignore_errors=True,
                          )
        except:
            pass

        os.mkdir(system_out_dir.resolve())
        os.mkdir(system_copied_dir.resolve())
        os.mkdir(system_aligned_dir.resolve())
        os.mkdir(system_processed_dir.resolve())
    return


def get_reference_dataset_dtag(datasets):
    reference_dtag = list(datasets.keys())[0]
    return reference_dtag


def copy_datasets(datasets,
                  target_dir,
                  ):
    copy_datasets_tasks = [CopyDataset(dtag=dtag,
                                       dataset_path=datasets[dtag],
                                       output_path=target_dir / dtag,
                                       )
                           for dtag
                           in datasets
                           ]

    luigi.build(copy_datasets_tasks,
                workers=20,
                local_scheduler=True,
                )


#
def align_datasets(reference_dtag,
                   dataset_paths,
                   reference_pdb_path,
                   out_dir,
                   min_res,
                   structure_factors,
                   ):
    align_dataset_tasks = [dataset_clustering.luigi_lib.AlignMapToReference(dtag=dtag,
                                                                            reference_dtag=reference_dtag,
                                                                            dataset_path=dataset_paths[dtag],
                                                                            reference_pdb_path=reference_pdb_path,
                                                                            output_path=out_dir / dtag,
                                                                            min_res=min_res,
                                                                            structure_factors=structure_factors,
                                                                            )
                           for dtag
                           in copied_datasets
                           ]
    luigi.build(align_dataset_tasks,
                workers=20,
                local_scheduler=True,
                )


class Loader:
    def __init__(self, d, min_res=None, grid_params=None):
        self.d = d
        self.min_res = min_res
        self.grid_params = grid_params

    def __call__(self):
        return MCDXMap.xmap_from_dataset(self.d,
                                         resolution=clipper_python.Resolution(self.min_res),
                                         grid_params=self.grid_params,
                                         )


def load_xmap(dataset, min_res):
    xmap = Loader(dataset, min_res=min_res)()
    return xmap


def dispatch(func_dict
             ):
    results = {}

    for i, func in func_dict.items():
        results[i] = func()

    return results


def get_xmaps(ccp4_map_paths):
    xmaps = dispatch({dtag: partial(load_ccp4_map,
                                    ccp4_map_path["path"],
                                    )
                      for dtag, ccp4_map_path
                      in ccp4_map_paths.items()
                      }
                     )
    return xmaps


# def get_xmaps(datasets):
#     xmaps = dispatch({dtag: partial(load_xmap,
#                                     dataset,
#                                     min_res=min_res,
#                                     )
#                       for dtag, dataset
#                       in datasets.datasets.items()
#                       }
#                      )
#     return xmaps


def get_min_res(datasets):
    ress = [res_from_mtz_file(d["mtz_path"]).limit
            for dtag, d
            in datasets.items()
            ]
    min_res = max(ress)
    return min_res


def xmap_to_np(xmap):
    return xmap.export_numpy()


def get_xmaps_np(xmaps):
    xmaps_np = dispatch({dtag: partial(xmap_to_np,
                                       xmap,
                                       )
                         for dtag, xmap
                         in xmaps.items()
                         }
                        )

    return xmaps_np


def embed(numpy_maps):
    # Convert to sample by feature
    sample_by_feature = np.vstack([np_map.flatten()
                                   for dtag, np_map
                                   in numpy_maps.items()
                                   ]
                                  )

    # Reduce dimension by PCA
    # transform = FastICA(n_components=3)
    # transform = FactorAnalysis(n_components=4)
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
                                min_cluster_size=3,
                                )
    # labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    clusterer.fit(xmap_embedding.astype(np.float64))

    return clusterer


def cluster(xmap_embedding):
    clustering = cluster_embedding(xmap_embedding)
    return clustering


def get_centroid_maps(xmaps_np,
                      clustering,
                      ):
    clustered_xmaps = clustering.labels_
    dtags = np.array(list(xmaps_np.keys()))
    outlier_scores = clustering.outlier_scores_

    centroid_maps = {}
    for cluster_num in np.unique(clustered_xmaps):
        print("\tProcessing cluster number {}".format(cluster_num))
        if cluster_num == -1:
            continue

        cluster_dtags = dtags[clustered_xmaps == cluster_num]
        cluster_outlier_scores = outlier_scores[clustered_xmaps == cluster_num]
        centroid_dtag = cluster_dtags[np.argmin(cluster_outlier_scores)]
        print("component {} centroid dtag is {}".format(cluster_num, centroid_dtag))
        centroid_maps[cluster_num] = xmaps_np[centroid_dtag]

    return centroid_maps


def embed_centroids(xmaps_np,
                    centroid_maps,
                    ):
    components = []
    for cluster_num, centroid_map in centroid_maps.items():
        component = centroid_map.flatten().reshape(-1, 1)
        print(component.shape)
        components.append(component)

    design_matrix = np.hstack(components)

    regression_components = {}
    for dtag, xmap in xmaps_np.items():
        target = xmap.reshape(-1, 1)

        # print(target.shape, design_matrix.shape)

        fit_model = OLS(target, design_matrix).fit()
        # print(fit_model)
        # print(dir(fit_model))

        regression_components[dtag] = fit_model.params
        # print(regression_components)

    components_matrix = np.vstack(regression_components.values())
    # print(components_matrix.shape)

    return components_matrix


def embed_pca_inliers(clustering,
                      xmaps_np,
                      ):
    sample_by_feature = np.vstack([np_map.flatten()
                                   for dtag, np_map
                                   in xmaps_np.items()
                                   ]
                                  )

    num_pca_components = len(np.unique(clustering[clustering != -1]))

    non_outlying_maps = sample_by_feature[clustering != -1]

    components = np.zeros((len(xmaps_np),
                           num_pca_components,
                           ),
                          )

    pca = FastICA(n_components=num_pca_components)
    embedding = pca.fit_transform(non_outlying_maps)

    components[clustering != -1] = embedding

    return components


def embed_mixture_variational(xmaps_np,
                              n_components,
                              ):
    sample_by_feature = np.vstack([np_map.flatten()
                                   for dtag, np_map
                                   in xmaps_np.items()
                                   ]
                                  )

    # mixture = BayesianGaussianMixture()

    begin = time.time()

    pca = PCA(n_components=50)
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    print("shape is: {}".format(sample_by_feature_pca.shape))

    mixture = BayesianGaussianMixture(n_components,
                                      covariance_type="spherical",
                                      verbose=10,
                                      verbose_interval=2,
                                      )
    mixture.fit(sample_by_feature_pca)

    finish = time.time()

    print(mixture)

    print("Finished in {}".format(finish - begin))

    print(mixture.predict(sample_by_feature_pca))

    print(mixture.weights_)

    clusters = mixture.predict(sample_by_feature_pca)

    probabilities = mixture.score_samples(sample_by_feature_pca)

    return mixture, pca, clusters, probabilities


def embed_mixture(xmaps_np,
                  n_components,
                  ):
    sample_by_feature = np.vstack([np_map.flatten()
                                   for dtag, np_map
                                   in xmaps_np.items()
                                   ]
                                  )

    # mixture = BayesianGaussianMixture()

    begin = time.time()

    pca = PCA(n_components=50)
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    print("shape is: {}".format(sample_by_feature_pca.shape))

    # mixture = GaussianMixture(n_components,
    #                           covariance_type="spherical",
    #                           verbose=10,
    #                           verbose_interval=2,
    #                           )
    mixture = GaussianMixture(n_components,
                              covariance_type="spherical",
                              verbose=10,
                              verbose_interval=2,
                              )
    mixture.fit(sample_by_feature_pca)

    finish = time.time()

    print(mixture)

    print(mixture.bic(sample_by_feature_pca))

    print("Finished in {}".format(finish - begin))

    print(mixture.predict(sample_by_feature_pca))

    return mixture, pca


def get_basis_maps(xmaps_np,
                   clusters,
                   probabilities,
                   ):
    dtags = np.array(list(xmaps_np.keys()))
    print(dtags.shape)
    xmaps = list(xmaps_np.values())

    basis_maps = {}
    for cluster in np.unique(clusters):
        cluster_probs = probabilities[clusters == cluster]
        cluster_dtags = dtags[clusters == cluster]
        print("Probabilities of cluster opints are: {}".format(cluster_probs))
        representative_index = np.argmax(cluster_probs)

        print(cluster, cluster_dtags[representative_index])
        basis_maps[cluster] = xmaps_np[cluster_dtags[representative_index]]

    return basis_maps


def mtzs_to_ccp4s(reference_pdb_path,
                  target_dir,
                  min_res,
                  ):
    align_dataset_tasks = [dataset_clustering.luigi_lib.MTZToCCP4(dtag=dtag,
                                                                  reference_dtag=reference_dtag,
                                                                  dataset_path=copied_datasets[dtag],
                                                                  reference_pdb_path=reference_pdb_path,
                                                                  output_path=target_dir / dtag,
                                                                  min_res=min_res,
                                                                  structure_factors=args.structure_factors,
                                                                  )
                           for dtag
                           in copied_datasets
                           ]

    luigi.build(align_dataset_tasks,
                workers=20,
                local_scheduler=True,
                )


def output_dataframe(cluster_df,
                     out_path,
                     ):
    output_labeled_embedding_csv(cluster_df,
                                 str(out_path),
                                 )


def make_mean_maps(xmaps_np,
                   cluster_df,
                   ):
    # exemplars = clustering.exemplars_
    # clustered_xmaps = clustering.labels_
    # outlier_scores = clustering.outlier_scores_
    # print("Exemplars: {}".format(exemplars))
    # print("Outlier scores: {}".format(outlier_scores))
    # print("Labels: {}".format(clustered_xmaps))
    #
    # dtags = np.array(list(xmaps_np.keys()))
    # cluster_exemplars = {}
    # for cluster_num in np.unique(clustered_xmaps):
    #     if cluster_num == -1:
    #         continue
    #     cluster_dtags = dtags[clustered_xmaps == cluster_num]
    #     cluster_outlier_scores = outlier_scores[clustered_xmaps == cluster_num]
    #     cluster_exemplars[cluster_num] = cluster_dtags[np.argmin(cluster_outlier_scores)]
        # print("Outlier scores dict: {}".format({dtag: outlier_score
        #                                         for dtag, outlier_score
        #                                         in zip(list(cluster_dtags),
        #                                                list(cluster_outlier_scores),
        #                                                )
        #                                         }
        #                                        )
        #       )

    map_clusters = get_map_clusters(cluster_df,
                                    xmaps_np,
                                    )

    # print(map_clusters)

    mean_maps_np = dispatch({i: partial(make_mean_map,
                                        map_cluster,
                                        )
                             for i, map_cluster
                             in map_clusters.items()
                             }
                            )
    return mean_maps_np


def make_component_maps(transform,
                        template_nxmap,
                        ):
    n_components = transform.n_components

    basis_vectors = np.eye(n_components)

    component_maps = {}

    for i in range(basis_vectors.shape[0]):
        unit_vec = basis_vectors[:, i].reshape(1, -1)

        component_maps[i] = transform.inverse_transform(unit_vec).reshape(template_nxmap.export_numpy().shape)
        print(component_maps[i].shape)

    return component_maps


def make_mixture_maps(mixture,
                      template_nxmap,
                      transform,
                      ):
    mixture_means_array = mixture.means_

    mixture_means_list = np.vsplit(mixture_means_array,
                                   mixture_means_array.shape[0],
                                   )

    mixture_maps = {}

    map_shape = template_nxmap.export_numpy().shape

    for i, mixture_mean_map in enumerate(mixture_means_list):
        mixture_maps[i] = transform.inverse_transform(mixture_mean_map).reshape(map_shape)

    return mixture_maps


def get_template_nxmap(ccp4_map_paths):
    template_nxmap = load_ccp4_map(list(ccp4_map_paths.values())[0]["path"])
    return template_nxmap


def get_cluster_df(dataset, xmap_embedding, clustered_xmaps):
    records = []

    for i, key in enumerate(dataset):
        record = {"dtag": key,
                  "cluster": clustered_xmaps[i]
                  }
        for j in range(xmap_embedding.shape[1]):
            record["component_{}".format(j)] = xmap_embedding[i, j]
        records.append(record)

    df = pd.DataFrame(records)

    return df


def make_dataframe(xmaps_np,
                   xmap_embedding,
                   clustered_xmaps,
                   ):
    cluster_df = get_cluster_df(xmaps_np,
                                xmap_embedding,
                                clustered_xmaps,
                                )
    # cluster_df_summary(cluster_df)
    return cluster_df


def output_mean_maps(mean_maps_np,
                     template_nxmap,
                     out_dir,
                     ):
    for cluster_num, mean_map_np in mean_maps_np.items():
        dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
                                                               mean_map_np,
                                                               out_dir / "{}.ccp4".format(cluster_num),
                                                               )


def output_component_maps(component_maps,
                          template_nxmap,
                          out_dir,
                          ):
    for i, component_map_np in component_maps.items():
        dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
                                                               component_map_np,
                                                               out_dir / "component_{}.ccp4".format(i),
                                                               )


def output_html_graph(df,
                      out_path,
                      ):
    df["cluster"] = df["cluster"].apply(str)

    # Get cds
    cds = ColumnDataSource(df)

    # # use whatever palette you want...
    palette = d3['Category20'][(len(df['cluster'].unique()) % 19) + 2]
    color_map = bmo.CategoricalColorMapper(factors=df['cluster'].unique(),
                                           palette=palette)

    # Define tooltipts
    TOOLTIPS = [
        ("dtag", "@dtag"),
        ("(component_0,component_1)", "($x, $y)"),
        ("cluster", "@cluster")
    ]

    # Gen figure
    p = figure(plot_width=800,
               plot_height=800,
               tooltips=TOOLTIPS,
               title="Mouse over the dots")

    # Plot data
    p.circle('component_0',
             'component_1',
             color={'field': 'cluster', 'transform': color_map}, size=10, source=cds)

    # Save figure
    save(p,
         str(out_path),
         )


if __name__ == "__main__":
    # get_args(): IO<cla> -> Args
    args = get_args()

    # get_datasets: Path, Regex, Regex -> Map<ID, Dataset>
    datasets = get_datasets(args.root_path,
                            args.mtz_regex,
                            args.pdb_regex,
                            )

    # setup the output
    setup_output(args.out_dir)

    # get_min_res
    min_res = get_min_res(datasets)

    # truncate_datasets: Map<ID, Dataset> -> Map<ID, Dataset>
    truncated_datasets = truncate_datasets(datasets,
                                           min_res,
                                           )

    # get_reference_dataset_dtag: Map<ID, Dataset> -> ID
    reference_dtag = get_reference_dataset_dtag(truncated_datasets)

    # copy_datasets: Map<Id, Dataset> -> IO<Dataset>
    copy_datasets(truncated_datasets,
                  args.out_dir / "copied",
                  )

    # get_datasets: Path, Regex, Regex -> Map<ID, Dataset>
    copied_datasets = get_datasets(args.out_dir / "copied",
                                   "*.mtz",
                                   "*.pdb",
                                   )

    # normalise_stucture:
    print(copied_datasets)
    normalise_structure(copied_datasets[reference_dtag]["pdb_path"],
                        reference_dtag,
                        output_path=args.out_dir,
                        )

    # align:
    # mtzs_to_ccp4s(args.out_dir / "{}_normalised.pdb".format(reference_dtag),
    #               args.out_dir / "maps",
    #               min_res,
    #               )
    align_datasets(reference_dtag,
                   copied_datasets,
                   args.out_dir / "{}_normalised.pdb".format(reference_dtag),
                   args.out_dir / "maps",
                   min_res,
                   args.structure_factors,
                   )

    #
    ccp4_map_paths = parse_pandda_input_for_regex(args.out_dir / "maps",
                                                  "*.ccp4",
                                                  )

    #
    template_nxmap = get_template_nxmap(ccp4_map_paths)

    # get_xmaps:
    xmaps = get_xmaps(ccp4_map_paths)

    # get_arrays:
    xmaps_np = get_xmaps_np(xmaps)

    # find clusters
    embedding = embed(xmaps_np)
    clustering = cluster(embedding)

    #
    df_tsne = make_dataframe(xmaps_np,
                             embedding,
                             clustering.labels_,
                             )

    # mixture = embed_mixture(xmaps_np,
    #                         6,
    #                         )

    # mixture_mean_maps = make_mixture_maps(mixture,
    #                                       template_nxmap,
    #                                       )

    # for j, mixture_map in mixture_mean_maps.items():
    #     dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
    #                                                            mixture_map,
    #                                                            args.out_dir / "component_{}_{}.ccp4".format(i, j),
    #                                                            )

    if args.method == "mixture":

        mixture, transform, clusters, probabilities = embed_mixture_variational(xmaps_np,
                                                                                12,
                                                                                )



        mixture_mean_maps = make_mixture_maps(mixture,
                                              template_nxmap,
                                              transform,
                                              )

        for j, mixture_map in mixture_mean_maps.items():
            dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
                                                                   mixture_map,
                                                                   args.out_dir / "component_{}.ccp4".format(j),
                                                                   )

        cluster_df = make_dataframe(xmaps_np,
                                    embedding,
                                    clusters,
                                    )

        output_dataframe(cluster_df,
                         Path(args.out_dir) / "cluster_mixture.csv",
                         )

        output_html_graph(cluster_df,
                          args.out_dir / "cluster_mixture.html",
                          )

        # make_mean_maps:
        mean_maps = make_mean_maps(xmaps_np,
                                   cluster_df,
                                   )

        output_mean_maps(mean_maps,
                         template_nxmap,
                         args.out_dir,
                         )

        basis_maps = get_basis_maps(xmaps_np,
                                    clusters,
                                    probabilities,
                                    )

        # embed_arrays:
        embedding_centroids = embed_centroids(xmaps_np,
                                              basis_maps,
                                              )

        # cluster_embeddings:
        clustering_centroids = cluster(embedding_centroids)

        # make_dataframe:
        cluster_components_df = make_dataframe(xmaps_np,
                                    embedding_centroids,
                                    clustering_centroids.labels_,
                                    )

        # output:
        output_dataframe(cluster_components_df,
                         Path(args.out_dir) / "cluster_components.csv",
                         )

        output_html_graph(cluster_components_df,
                          args.out_dir / "cluster_components.html",
                          )



        # for i in range(12):
        #     mixture, transform = embed_mixture(xmaps_np,
        #                                        i + 1,
        #                                        )
        #
        #     mixture_mean_maps = make_mixture_maps(mixture,
        #                                           template_nxmap,
        #                                           transform,
        #                                           )
        #
        #     for j, mixture_map in mixture_mean_maps.items():
        #         dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
        #                                                                mixture_map,
        #                                                                args.out_dir / "component_{}_{}.ccp4".format(i,
        #                                                                                                             j),
        #                                                                )

    if args.method == "pca_inliers":
        embedding_pca_inliers = embed_pca_inliers(clustering.labels_,
                                                  xmaps_np,
                                                  )
        clustering_pca_inliers = cluster(embedding_pca_inliers)

        cluster_df = make_dataframe(xmaps_np,
                                    embedding_pca_inliers,
                                    clustering_pca_inliers.labels_,
                                    )

        output_dataframe(cluster_df,
                         Path(args.out_dir) / "cluster_pca_inliers.csv",
                         )

        output_html_graph(cluster_df,
                          args.out_dir / "cluster_pca_inliers.html",
                          )

    if args.method == "centroids":
        # find centroids
        centroid_maps = get_centroid_maps(xmaps_np,
                                          clustering,
                                          )

        # embed_arrays:
        embedding_centroids = embed_centroids(xmaps_np,
                                              centroid_maps,
                                              )

        # cluster_embeddings:
        clustering_centroids = cluster(embedding_centroids)

        # make_dataframe:
        cluster_df = make_dataframe(xmaps_np,
                                    embedding_centroids,
                                    clustering_centroids.labels_,
                                    )

        # output:
        output_dataframe(cluster_df,
                         Path(args.out_dir) / "labelled_embedding_components.csv",
                         )

        output_html_graph(cluster_df,
                          args.out_dir / "cluster_components.html",
                          )

    output_dataframe(df_tsne,
                     Path(args.out_dir) / "labelled_embedding_tsne.csv",
                     )
    output_html_graph(df_tsne,
                      args.out_dir / "cluster_tsne.html"
                      )

    # make_mean_maps:
    # mean_maps = make_mean_maps(clustering_centroids,
    #                            xmaps_np,
    #                            cluster_df,
    #                            )

    # make_component_maps:
    # component_maps = make_component_maps(transform,
    #                                      template_nxmap,
    #                                      )

    # output_mean_maps(mean_maps,
    #                  template_nxmap,
    #                  args.out_dir,
    #                  )
    # output_component_maps(component_maps,
    #                       template_nxmap,
    #                       args.out_dir,
    #                       )

    # cluster_df["cluster"] = df_tsne["cluster"]
    # output_html_graph(cluster_df,
    #                   args.out_dir / "cluster_components_tsne_labels.html",
    #                   )
