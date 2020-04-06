# Regular Imports
import os, sys, configparser
import time
from collections import OrderedDict
import argparse
import pathlib as p
import pickle



# Scientific imports
import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use('agg')
mpl.interactive(False)
from matplotlib import pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize': (11.7, 8.27)})

import multiprocessing

import joblib
from joblib.externals.loky import set_loky_pickler

from scipy.stats import mode

import hdbscan

from Bio import PDB

from Bio.PDB.vectors import Vector, rotmat

from Bio.PDB import PDBIO

from MDAnalysis.analysis import align

# clipper
import clipper_python
from clipper_python import Resolution

# mdc3
import mdc3
from mdc3.types.datasets import MultiCrystalDataset
from mdc3.types.real_space import (MCDXMap,
                                   xmap_to_numpy_cartesian_axis,
                                   xmap_from_dataset,
                                   xmap_to_numpy_crystalographic_axis,
                                   )
from mdc3.functions.alignment import align

# local
import dataset_clustering
from dataset_clustering.real_space_alignment import align_xmap_np
from dataset_clustering.clustering import (embed_xmaps,
                                           cluster_embedding,
                                           )

from dataset_clustering.data_handling import (get_cluster_df,
                                              output_labeled_embedding_csv,
                                              get_map_clusters,
                                              )

from dataset_clustering.xmap_utils import (make_mean_map,
                                           output_mean_map,
                                           output_mean_nxmap,
                                           ccp4_path_to_np,
                                           load_ccp4_map,
                                           )


def get_args():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("-c", "--config_path",
                        type=str,
                        default="config.ini",
                        help="Path to the config file with uncommon options"
                        )
    parser.add_argument("-d", "--data_dirs",
                        type=str,
                        help="The directory dir such that dir//<dataset_names>//<pdbs and mtzs>",
                        required=True
                        )
    parser.add_argument("-o", "--out_dir",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )
    parser.add_argument("-n", "--num_datasets",
                        type=str,
                        help="The number of datasets to use in a random sample",
                        required=True
                        )
    parser.add_argument("-p", "--processor",
                        type=str,
                        help="The processing method to use",
                        default="map",
                        )

    return parser


def align_structures(ref_structure, mobile_structure, mobile_xmap):

    # translation_mobile_to_ref = ref_structure.atoms.center_of_mass() - mobile_structure.atoms.center_of_mass()
    #
    #
    # mobile0 = mobile_structure.select_atoms('name CA').positions - mobile_structure.atoms.center_of_mass()
    # ref0 = ref_structure.select_atoms('name CA').positions - ref_structure.atoms.center_of_mass()
    # rotation_mobile_to_ref, rmsd = align.rotation_matrix(mobile0, ref0)

    ref_atoms = []
    alt_atoms = []
    for (ref_model, alt_model) in zip(ref_structure, mobile_structure):
        for (ref_chain, alt_chain) in zip(ref_model, alt_model):
            for ref_res, alt_res in zip(ref_chain, alt_chain):

                # CA = alpha carbon
                try:
                # print("\tModel: {}".format(ref_model))
                # print("\tChain: {}".format(ref_chain))
                # print("\tResidue: {}".format(ref_res))

                # print(ref_res)
                # print(dir(ref_res))
                # print([x for x in ref_res.get_atoms()])
                    ref_atoms.append(ref_res['CA'])
                    alt_atoms.append(alt_res['CA'])
                except:
                    pass

    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms,
                            alt_atoms,
                            )

    translation_mobile_to_ref = super_imposer.rotran[1]
    rotation_mobile_to_ref = super_imposer.rotran[0]

    # print("\tTranslation is: {}".format(translation_mobile_to_ref))
    # print("\tRotation is: {}".format(rotation_mobile_to_ref))
    # xmap_np = interpolate_uniform_grid(mobile_xmap,
    #                                    translation_mobile_to_ref,
    #                                    np.transpose(rotation_mobile_to_ref),
    #                                    )

    rtop = clipper_python.RTop_orth(clipper_python.Mat33_double(np.transpose(rotation_mobile_to_ref)),
                                    clipper_python.Vec3_double(translation_mobile_to_ref),
                                    )

    xmap_new = clipper_python.Xmap_float(mobile_xmap.xmap.spacegroup,
                                         mobile_xmap.xmap.cell,
                                         mobile_xmap.xmap.grid_sampling,
                                         )

    clipper_python.rotate_translate(mobile_xmap.xmap,
                                    xmap_new,
                                    rtop,
                                    )

    return xmap_new.export_numpy()

#
# def align(ref_structure, mobile_structure):
#
#     # translation_mobile_to_ref = ref_structure.atoms.center_of_mass() - mobile_structure.atoms.center_of_mass()
#     #
#     #
#     # mobile0 = mobile_structure.select_atoms('name CA').positions - mobile_structure.atoms.center_of_mass()
#     # ref0 = ref_structure.select_atoms('name CA').positions - ref_structure.atoms.center_of_mass()
#     # rotation_mobile_to_ref, rmsd = align.rotation_matrix(mobile0, ref0)
#
#     ref_atoms = []
#     alt_atoms = []
#     for (ref_model, alt_model) in zip(ref_structure, mobile_structure):
#         for (ref_chain, alt_chain) in zip(ref_model, alt_model):
#             for ref_res, alt_res in zip(ref_chain, alt_chain):
#
#                 # CA = alpha carbon
#                 try:
#                     # print("\tModel: {}".format(ref_model))
#                     # print("\tChain: {}".format(ref_chain))
#                     # print("\tResidue: {}".format(ref_res))
#
#                     # print(ref_res)
#                     # print(dir(ref_res))
#                     # print([x for x in ref_res.get_atoms()])
#                     ref_atoms.append(ref_res['CA'])
#                     alt_atoms.append(alt_res['CA'])
#                 except:
#                     pass
#
#     super_imposer = PDB.Superimposer()
#     super_imposer.set_atoms(ref_atoms,
#                             alt_atoms,
#                             )
#
#
#     return super_imposer


def wrap_call(x):
    return x()


# class StructureAligner:
#     def __init__(self, ref_structure, mobile_structure, mobile_xmap):
#         self.mobile_structure = mobile_structure
#         self.ref_structure = ref_structure
#         self.mobile_xmap = mobile_xmap
#
#     def __call__(self):
#         return align_structures(self.ref_structure,
#                                 self.mobile_structure,
#                                 self.mobile_xmap,
#                                 )


def get_min_res(mcd: MultiCrystalDataset) -> float:

    ress = [d.reflections.hkl_info.resolution.limit
            for dtag, d
            in mcd.datasets.items()
            ]
    return max(ress)


def mapdict(func, dictionary, executor):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # results = executor(func,
    #                    values,
    #                    )

    results = [func(value) for value in values]

    return {key: result for key, result in zip(keys, results)}


def wrap_xmap_from_dataset(dataset, resolution=Resolution(0)):
    return MCDXMap.xmap_from_dataset(dataset, resolution=resolution)


def filter_on_grid(xmaps_np):
    new_xmaps_np = {}
    shape = xmaps_np[list(xmaps_np.keys())[0]].shape
    for dtag, xmap_np in xmaps_np.items():
        if xmap_np.shape != shape:
            print("Filtering dataset: {}".format(dtag))
            print("\tDifferent array shape due to unit cell params! Should have {}, found {}".format(shape,
                                                                                                     xmap_np.shape,
                                                                                                     )
                  )
        else:
            new_xmaps_np[dtag] = xmap_np

    return new_xmaps_np


class ExecutorJoblib:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def __call__(self, func, values):
        results = joblib.Parallel(n_jobs=self.n_jobs,
                                  backend="loky",
                                  verbose=9,
                                  )(joblib.delayed(joblib.wrap_non_picklable_objects(func))(value)
                                    for value
                                    in values
                                    )

        # return [x[0] for x in results]
        return results


class Executormultiprocessing:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def __call__(self, func, values):

        with multiprocessing.Pool(self.n_jobs) as p:
            results = p.map(func, values)

        return results


class ExecutorMap:
    def __init__(self):
        pass

    def __call__(self, func, values):
        results = map(func,
                      values,
                      )
        return results


class Loader:
    def __init__(self, min_res=None, grid_params=None):
        self.min_res = min_res
        self.grid_params = grid_params

    def __call__(self, d):
        return MCDXMap.xmap_from_dataset(d,
                                         resolution=clipper_python.Resolution(self.min_res),
                                         grid_params=self.grid_params,
                                         )


class Aligner:
    def __init__(self, static_map):
        self.static_map = static_map

    def __call__(self, xmap_np):
        return align_xmap_np(self.static_map,
                             xmap_np,
                             )


class Rescaler:
    def __init__(self):
        pass

    def __call__(self, xmap_np):
        return xmap_np / np.std(xmap_np)


class ImageAligner:
    def __init__(self, static_image, moving_image):
        self.static_image = static_image
        self.moving_image = moving_image

    def __call__(self):
        return align_xmap_np(self.static_image,
                             self.moving_image,
                             )


class Sampler:
    def __init__(self,
                 xmap,
                 grid_params,
                 translation_mobile_to_ref=[0, 0, 0],
                 rotation_mobile_to_ref=np.eye(3),
                 ):
        self.xmap = xmap
        self.grid_params = grid_params

        self.translation_mobile_to_ref = translation_mobile_to_ref
        self.rotation_mobile_to_ref = rotation_mobile_to_ref

    def __call__(self):

        rot = clipper_python.Mat33_double(self.rotation_mobile_to_ref)
        trans = clipper_python.Vec3_double(self.translation_mobile_to_ref[0],
                                           self.translation_mobile_to_ref[1],
                                           self.translation_mobile_to_ref[2],
                                           )

        rtop = clipper_python.RTop_orth(rot,
                                        trans,
                                        )

        # Generate the clipper grid
        grid = clipper_python.Grid(self.grid_params[0],
                                   self.grid_params[1],
                                   self.grid_params[2],
                                   )

        # Define nxmap from the clipper grid and rotation-translation operator
        nxmap = clipper_python.NXmap_float(grid,
                                           rtop,
                                           )

        # Interpolate the Xmap onto the clipper nxmap
        clipper_python.interpolate(nxmap, self.xmap.xmap)

        return mdc3.types.real_space.MCDNXMap(nxmap)


class StructureAligner:
    def __init__(self, reference_structure, moving_structure):
        self.reference_structure = reference_structure
        self.moving_structure = moving_structure

    def __call__(self):
        return align(ref_structure=self.reference_structure,
                     mobile_structure=self.moving_structure,
                     ).rotran


def postcondtion_scaling(xmaps_np):
    for dtag, xmap_np in xmaps_np.items():
        if not np.isclose(np.std(xmap_np), 1.0, atol=0.05):
            raise Exception("Std should be near 1, instead is: {}".format(np.std(xmap_np)))


    print("\tSUCESS: All rescaling postconditions passed")


def postcondition_alignment(xmaps_np):
    # assert type
    for dtag, xmap_np in xmaps_np.items():
        if type(dtag) != str:
            raise Exception("Dictionary key not a string, is {}".format(type(dtag)))
        if type(xmap_np) != np.ndarray:
            raise Exception("Alignment output is not an ndarray, is {}".format(type(xmap_np)))

    # assert shape
    shape = xmaps_np[list(xmaps_np.keys())[0]].shape
    for dtag, xmap_np in xmaps_np.items():
        if xmap_np.shape != shape:
            raise Exception("Xmap shape should be {}, is actually {}".format(shape,
                                                                             xmap_np.shape,
                                                                             )
                            )

    # asser non-zero
    for dtag, xmap_np in xmaps_np.items():
        if not xmap_np.any():
            raise Exception("Xmap seems to be all zeros, which it shouldn't be")

    print("Alignment Postconditions met!")


def postcondition_loading(xmaps):
    # assert type
    for dtag, xmap in xmaps.items():
        if type(dtag) != str:
            raise Exception("Dictionary key not a string, is {}".format(type(dtag)))
        if type(xmap) != mdc3.types.real_space.MCDXMap:
            raise Exception("Alignment output is not an ndarray, is {}".format(type(xmap)))

    print("Loading postconditions met!")


def cluster_df_summary(cluster_df):
    print("Cluster dataframe summary")
    print("\tClusters: {}".format(cluster_df["cluster"].unique()))
    print("\tNum clusters: {}".format(len(cluster_df["cluster"].unique())))
    print("\tNum datasets clustered: {}".format(len(cluster_df)))
    print("\tNumber of outliers: {}".format(len(cluster_df[cluster_df["cluster"] == -1])))
    print("\tNumber of datasets by cluster: {}".format({i: len(cluster_df[cluster_df["cluster"] == i])
                                                        for i
                                                        in cluster_df["cluster"].unique()
                                                        }
                                                       )
          )


def truncate_on_res(dataset, q=0.95):
    ress = {dtag: d.reflections.hkl_info.resolution.limit
            for dtag, d
            in dataset.datasets.items()
            }

    cutoff = np.quantile(list(ress.values()), q)

    new_datasets = {}
    for dtag, res in ress:
        if res < cutoff:
            new_datasets[dtag] = dataset.datasets[dtag]

    dataset.datasets = new_datasets

    return dataset


# def summarise_truncation(old_dataset, new_dataset):
#     for dtag, d in old_dataset:
#         try:
#             new_dataset[dtag] =
#         except Exception as e:
#             # print("\t" + str(e))
#             print("\tDataset {} removed for being in the top 5% of the resolution range".format())


def cluster_unit_cell_params(dataset):

    sample_by_feature = get_unit_cell_params_as_sample_by_feature(dataset)

    clustering = hdbscan.HDBSCAN(allow_single_cluster=True,
                                 min_cluster_size=15,
                                 )
    clustering.fit(sample_by_feature)

    labels = {dtag: i for dtag, i in zip(dataset.datasets.keys(),
                                                             clustering.labels_,
                                                             )
              }

    print("\tExemplar cluster is: {}".format(clustering.exemplars_[0][0]))

    return labels


def truncate_dataset_on_clustering(dataset, labels):
    modal_cluster = mode(list(labels.values()),
                         axis=None,
                         )
    print("\tModal cluster is modal cluster: {}".format(modal_cluster.mode[0]))



    new_datasets = {}

    for dtag, d in dataset.datasets.items():
        # print("\t{}".format(modal_cluster.mode[0]))
        # print("\t{}".format(labels[dtag]))
        if labels[dtag] == modal_cluster.mode[0]:
            new_datasets[dtag] = d
        else:
            print("\tRemoving dataset {}: VERY DIFFERENT UNIT CELL PARAMETERS OF {}".format(dtag,
                                                                                            np.hstack((d.reflections.hkl_info.cell.dim,
                                                                                                      d.reflections.hkl_info.cell.angles,
                                                                                                      ),
                                                                                                      )
                                                                                            )
                  )
    dataset.datasets = new_datasets
    return dataset


def get_unit_cell_params_as_sample_by_feature(dataset):

    sample_by_features_list = []

    for dtag, d in dataset.datasets.items():
        dimensions = d.reflections.hkl_info.cell.dim
        angles = d.reflections.hkl_info.cell.angles
        dim_angles = np.hstack((dimensions, angles))

        sample_by_features_list.append(dim_angles)

    return np.array(sample_by_features_list)


def truncate_dataset_on_resolution(dataset, cutoff=3.0):

    new_datasets = {}

    print("\tCutoff resolution is: {}".format(cutoff))

    for dtag, d in dataset.datasets.items():
        # print("\t{}".format(modal_cluster.mode[0]))
        # print("\t{}".format(labels[dtag]))
        if d.reflections.hkl_info.resolution.limit < cutoff:
            new_datasets[dtag] = d
        else:
            print("\tRemoving dataset {}: VERY BAD RESOLUTION OF {}".format(dtag,
                                                                            d.reflections.hkl_info.resolution.limit,
                                                                            )
                  )


    dataset.datasets = new_datasets
    return dataset


def interpolate_uniform_grid(xmap,
                             translation_mobile_to_ref=(0, 0, 0),
                             rotation_mobile_to_ref=np.eye(3),
                             ):
    rot = clipper_python.Mat33_double(rotation_mobile_to_ref)
    trans = clipper_python.Vec3_double(translation_mobile_to_ref[0],
                                       translation_mobile_to_ref[1],
                                       translation_mobile_to_ref[2],
                                       )

    rtop = clipper_python.RTop_orth(rot,
                                    trans,
                                    )

    # Generate the clipper grid
    grid = clipper_python.Grid(100,
                               100,
                               100,
                               )

    # Define nxmap from the clipper grid and rotation-translation operator
    nxmap = clipper_python.NXmap_float(grid,
                                       rtop,
                                       )

    # Interpolate the Xmap onto the clipper nxmap
    # print("interpolating!")
    clipper_python.interpolate(nxmap, xmap.xmap)

    xmap_np = nxmap.export_numpy()

    return xmap_np


def nxmap_to_numpy(nxmap):
    return nxmap.nxmap.export_numpy()



########################################################################################################################
def cluster_datasets(args) -> None:
    # Get executor
    if args.processor == "joblib":
        set_loky_pickler("pickle")
        executor = ExecutorJoblib(20)
    elif args.processor == "map":
        executor = ExecutorMap()
    executor_map = ExecutorMap()

    print("defined executor")

    # Get Dataset
    print("Loading datasets")
    dataset = MultiCrystalDataset.mcd_from_pandda_input_dir(p.Path(args.data_dirs))
    # dataset.datasets = {key: value for key, value in list(dataset.datasets.items())[:150]}
    print("got datasets")

    print("\tBefore tuncation on res there are {} datasets".format(len(dataset.datasets)))
    dataset = truncate_dataset_on_resolution(dataset)
    print("\tAfter truncationt on res here are {} datasets".format(len(dataset.datasets)))
    #
    #
    # unit_cell_clustering_labels = cluster_unit_cell_params(dataset)
    # print("\tBefore tuncation there are {} datasets".format(len(dataset.datasets)))
    # dataset = truncate_dataset_on_clustering(dataset, unit_cell_clustering_labels)
    # print("\tAfter truncationt here are {} datasets".format(len(dataset.datasets)))

    # Select lowest res dataset
    min_res = get_min_res(dataset)
    # dataset = truncate_on_res(dataset)



    # d = dataset.datasets[list(dataset.datasets.keys())[0]]
    # xmap = MCDXMap.xmap_from_dataset(d,
    #                                  resolution=Resolution(min_res),
    #                                  )
    # print(xmap)
    # xmap_np = xmap_to_numpy_crystalographic_axis(xmap)
    # print(np.std(xmap_np))
    #
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(xmap, f)
    #
    # with open("test.pkl", "rb") as f:
    #     xmap_reloaded = pickle.load(f)
    #
    #
    # xmap_reloaded_np = xmap_to_numpy_crystalographic_axis(xmap_reloaded)
    # print(np.std(xmap_reloaded_np))

    #
    # exit()

    # Load xmaps
    print("Getting exmpas")
    # xmaps = mapdict(lambda d: MCDXMap.xmap_from_dataset(d,
    #                                                     resolution=Resolution(min_res),
    #                                                     ),
    #                 dataset.datasets,
    #                 executor,
    #                 )
    reference_dataset = dataset.datasets[list(dataset.datasets.keys())[0]]
    # print(dir(reference_dataset))
    reference_grid = clipper_python.Grid_sampling(reference_dataset.reflections.hkl_info.spacegroup,
                                                  reference_dataset.reflections.hkl_info.cell,
                                                  Resolution(min_res),
                                                  )
    grid_params = (reference_grid.nu,
                   reference_grid.nv,
                   reference_grid.nw,
                   )
    xmaps = mapdict(Loader(min_res=None,
                           grid_params=grid_params,
                           ),
                    dataset.datasets,
                    executor,
                    )
    # print("Xmap std is: {}".format(np.std(xmaps[list(xmaps.keys())[0]].xmap.export_numpy())))

    # Convert xmaps to np
    print("Converting xmaps to np")
    # xmaps_np = mapdict(xmap_to_numpy_crystalographic_axis,
    #                    xmaps,
    #                    executor,
    #                    )
    # postcondition_alignment(xmaps_np)
    # xmaps_np = mapdict(interpolate_uniform_grid,
    #                    xmaps,
    #                    executor,
    #                    )
    # print("\tXmap representative shape is: {}".format(xmaps_np[list(xmaps_np.keys())[0]].shape))

    static_structure = dataset.datasets[list(dataset.datasets.keys())[0]].structure.structure
    # print("Got static structure")
    aligners = {}
    for dtag, xmap in xmaps.items():
        aligners[dtag] = StructureAligner(static_structure,
                                          dataset.datasets[dtag].structure.structure,
                                          xmap,
                                          )

    xmaps_aligned = mapdict(wrap_call,
                            aligners,
                            executor,
                            )

    # Rescale
    rescaler = Rescaler()
    xmaps_np = mapdict(rescaler,
                       xmaps_aligned,
                       executor,
                       )
    postcondtion_scaling(xmaps_np)

    # Align xmaps
    print("aligning xmaps")
    # static_map = xmaps_np[list(xmaps_np.keys())[0]]
    # aligner = Aligner(static_map)
    # xmaps_aligned = mapdict(aligner,
    #                         xmaps_np,
    #                         executor,
    #                         )

    xmaps_np = filter_on_grid(xmaps_np)


    # Embed the xmaps into a latent space
    print("Dimension reducing")
    xmap_embedding = embed_xmaps(xmaps_np.values())

    # Cluster the embedding
    print("clustering")
    clustered_xmaps = cluster_embedding(xmap_embedding)

    # Make dataframe with cluster and position
    print("getting cluster dataframe")
    cluster_df = get_cluster_df(xmaps_np,
                                xmap_embedding,
                                clustered_xmaps,
                                )
    cluster_df_summary(cluster_df)

    # Get clusters
    print("associating xmaps with clusters")
    map_clusters = get_map_clusters(cluster_df,
                                    xmaps_aligned,
                                    )
    # print("Map clusters: {}".format(map_clusters))

    #  Make mean maps
    executor_seriel = ExecutorMap()
    mean_maps_np = mapdict(lambda x: make_mean_map(x),
                           map_clusters,
                           executor_seriel,
                           )
    # print("Mean maps mp: {}".format(mean_maps_np))

    # Output the mean maps
    print("Outputting mean maps")
    template_map = xmaps[list(xmaps.keys())[0]]
    # print(template_map)
    # cell = dataset.datasets[list(dataset.datasets.keys())[0]].reflections.hkl_info.cell
    cell = clipper_python.Cell(clipper_python.Cell_descr(100, 100, 100, np.pi / 2, np.pi / 2, np.pi / 2))
    for cluster_num, mean_map_np_list in mean_maps_np.items():
        output_mean_map(template_map,
                        mean_map_np_list,
                        p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
                        )
        # output_mean_nxmap(mean_map_np_list,
        #                   cell,
        #                   p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
        #                   )

    # Ouptut the csv
    print("Outputting csv")
    output_labeled_embedding_csv(cluster_df,
                                 str(p.Path(args.out_dir) / "labelled_embedding.csv"),
                                 )

    # Output the graph
    # output_cluster_graph(cluster_df, str(out_dir / "output_graph.png"))

    return cluster_df


def cluster_datasets_rsa(args) -> None:
    # Get executor
    if args.processor == "joblib":
        set_loky_pickler("pickle")
        executor = ExecutorJoblib(int(args.n_procs))
    elif args.processor == "map":
        executor = ExecutorMap()
    executor_map = ExecutorMap()

    print("defined executor")

    # Get Dataset
    print("Loading datasets")
    dataset = MultiCrystalDataset.mcd_from_pandda_input_dir(p.Path(args.data_dirs))
    # dataset.datasets = {key: value for key, value in list(dataset.datasets.items())[:150]}
    print("got datasets")

    print("\tBefore tuncation on res there are {} datasets".format(len(dataset.datasets)))
    dataset = truncate_dataset_on_resolution(dataset)
    print("\tAfter truncationt on res here are {} datasets".format(len(dataset.datasets)))

    # Select lowest res dataset
    min_res = get_min_res(dataset)

    # Load xmaps
    print("Getting exmpas")
    xmaps = mapdict(Loader(min_res=min_res),
                    dataset.datasets,
                    executor,
                    )

    # Align models to ref
    reference_dtag = list(dataset.datasets.keys())[0]
    reference_model = dataset.datasets[reference_dtag].structure.structure

    # Get model alignmetns
    aligners = {}
    for dtag, d in dataset.datasets.items():
        # print("\tAligning {} to {}".format(dtag, reference_dtag))
        aligners[dtag] = StructureAligner(reference_model,
                                          d.structure.structure,
                                          )

    alignments = mapdict(wrap_call,
                         aligners,
                         executor_map,
                         )

    # Sample Xmaps uniformly
    # grid_params = [50,50,50]
    reference_cell = xmaps[reference_dtag].xmap.cell
    grid_params = (int(reference_cell.a),
                   int(reference_cell.b),
                   int(reference_cell.c),
                   )
    print("Grid params are: {}".format(grid_params))
    samplers = {}
    for dtag, xmap in xmaps.items():
        rtop = alignments[dtag]
        mobile_to_ref_translation = rtop[1]
        mobile_to_ref_rotation = rtop[0]

        # print("\tDataset {} translation is: {}".format(dtag, mobile_to_ref_translation))
        # print("\tDataset {} rotation is: {}".format(dtag, mobile_to_ref_rotation.flatten()))


        samplers[dtag] = Sampler(xmap,
                                 grid_params,
                                 mobile_to_ref_translation,
                                 mobile_to_ref_rotation,
                                 )

    nxmaps = mapdict(wrap_call,
                     samplers,
                     executor,
                     )


    # Convert nxmaps to np
    print("Converting xmaps to np")
    xmaps_np = mapdict(nxmap_to_numpy,
                       nxmaps,
                       executor,
                       )


    # Rescale
    rescaler = Rescaler()
    xmaps_np = mapdict(rescaler,
                       xmaps_np,
                       executor,
                       )
    postcondtion_scaling(xmaps_np)

    # Align xmaps
    print("aligning xmaps")
    static_image = xmaps_np[list(xmaps_np.keys())[0]]
    aligners = {}
    for dtag, xmap_np in xmaps_np.items():
        aligners[dtag] = ImageAligner(static_image,
                                      xmap_np,
                                      )

    xmaps_aligned = mapdict(wrap_call,
                            aligners,
                            executor,
                            )

    # Embed the xmaps into a latent space
    print("Dimension reducing")
    xmap_embedding = embed_xmaps(xmaps_np.values())

    # Cluster the embedding
    print("clustering")
    clustered_xmaps = cluster_embedding(xmap_embedding)

    # Make dataframe with cluster and position
    print("getting cluster dataframe")
    cluster_df = get_cluster_df(xmaps_np,
                                xmap_embedding,
                                clustered_xmaps,
                                )
    cluster_df_summary(cluster_df)

    # Get clusters
    print("associating xmaps with clusters")
    map_clusters = get_map_clusters(cluster_df,
                                    xmaps_aligned,
                                    )

    #  Make mean maps
    executor_seriel = ExecutorMap()
    mean_maps_np = mapdict(lambda x: make_mean_map(x),
                           map_clusters,
                           executor_seriel,
                           )

    # Output the mean maps
    print("Outputting mean maps")
    template_map = xmaps[list(xmaps.keys())[0]]
    # print(template_map)
    # cell = dataset.datasets[list(dataset.datasets.keys())[0]].reflections.hkl_info.cell
    cell = clipper_python.Cell(clipper_python.Cell_descr(grid_params[0],
                                                         grid_params[1],
                                                         grid_params[2],
                                                         np.pi / 2,
                                                         np.pi / 2,
                                                         np.pi / 2,
                                                         )
                               )
    for cluster_num, mean_map_np in mean_maps_np.items():
        output_mean_nxmap(mean_map_np,
                          cell,
                          p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
                          grid_params,
                          )
        # output_mean_nxmap(mean_map_np_list,
        #                   cell,
        #                   p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
        #                   )

    # Ouptut the csv
    print("Outputting csv")
    output_labeled_embedding_csv(cluster_df,
                                 str(p.Path(args.out_dir) / "labelled_embedding.csv"),
                                 )

    # Output the graph
    # output_cluster_graph(cluster_df, str(out_dir / "output_graph.png"))

    return cluster_df


class LoaderCCP4:
    def __init__(self, path):
        self.path = path

    def __call__(self):
        return ccp4_path_to_np(self.path)


def cluster_datasets_luigi(ccp4_map_paths, out_dir, processor, n_procs, log) -> pd.DataFrame:

    log("\tSetting executor...")
    # Get executor
    if processor == "joblib":
        set_loky_pickler("pickle")
        executor = ExecutorJoblib(int(n_procs))
    elif processor == "map":
        executor = ExecutorMap()

    log("\tLoading xmaps to np...")
    # Load xmaps to np
    print("Getting exmpas")
    loaders = {}
    for dtag in ccp4_map_paths:
        loaders[dtag] = LoaderCCP4(ccp4_map_paths[dtag]["path"])
    xmaps_np = mapdict(wrap_call,
                       loaders,
                       executor,
                       )

    log("\tTruncating xmaps to same shape...")
    # Truncate to same shape
    dims = np.vstack([xmap.shape for xmap in xmaps_np.values()])
    print(dims)
    truncation_shape = np.min(dims,
                              axis=0,
                              )
    print("The truncation shape is: {}".format(truncation_shape))
    xmaps_np = {dtag: xmap_np[:truncation_shape[0],
                                :truncation_shape[1],
                                :truncation_shape[2],
                                ]
                          for dtag, xmap_np
                          in xmaps_np.items()
                          }

    # Embed the xmaps into a latent space
    log("\tEmbedding xmaps...")
    print("Dimension reducing")
    xmap_embedding = embed_xmaps(xmaps_np.values())

    # Cluster the embedding
    print("clustering")
    log("\tClustering...")
    # clustered_xmaps = cluster_embedding(xmap_embedding)
    clustering = cluster_embedding(xmap_embedding)

    exemplars = clustering.exemplars_
    clustered_xmaps = clustering.labels_
    outlier_scores = clustering.outlier_scores_
    print("Exemplars: {}".format(exemplars))
    print("Outlier scores: {}".format(outlier_scores))
    print("Labels: {}".format(clustered_xmaps))

    log("\tFinding exemplars...")
    dtags = np.array(list(xmaps_np.keys()))
    cluster_exemplars = {}
    for cluster_num in np.unique(clustered_xmaps):
        if cluster_num == -1:
            continue
        cluster_dtags = dtags[clustered_xmaps == cluster_num]
        cluster_outlier_scores = outlier_scores[clustered_xmaps == cluster_num]
        cluster_exemplars[cluster_num] = cluster_dtags[np.argmin(cluster_outlier_scores)]
        print("Outlier scores dict: {}".format({dtag: outlier_score
                                                for dtag, outlier_score
                                                in zip(list(cluster_dtags),
                                                       list(cluster_outlier_scores),
                                                       )
                                                }
                                               )
              )

    print("Cluster exemplars: {}".format(cluster_exemplars))

    # Make dataframe with cluster and position
    log("\tMaking dataframe...")
    print("getting cluster dataframe")
    cluster_df = get_cluster_df(xmaps_np,
                                xmap_embedding,
                                clustered_xmaps,
                                )
    cluster_df_summary(cluster_df)

    # Get clusters
    log("\tAssociating xmaps with clusters...")
    print("associating xmaps with clusters")
    map_clusters = get_map_clusters(cluster_df,
                                    xmaps_np,
                                    )

    #  Make mean maps
    log("\tMaking mean maps...")
    executor_seriel = ExecutorMap()
    mean_maps_np = mapdict(lambda x: make_mean_map(x),
                           map_clusters,
                           executor_seriel,
                           )

    # Output the mean maps
    log("\tOutputting mean maps...")
    print("Outputting mean maps")
    template_nxmap = load_ccp4_map(list(ccp4_map_paths.values())[0]["path"])
    for cluster_num, mean_map_np in mean_maps_np.items():
        dataset_clustering.xmap_utils.save_nxmap_from_template(template_nxmap,
                                                               mean_map_np,
                                                               p.Path(out_dir) / "{}.ccp4".format(cluster_num),
                                                               )

    # Ouptut the csv
    log("\tOutputting csvs...")
    print("Outputting csv")
    output_labeled_embedding_csv(cluster_df,
                                 str(p.Path(out_dir) / "labelled_embedding.csv"),
                                 )

    print("Cluster DF is: \n{}".format(cluster_df))

    return cluster_df


def cluster_datasets_tm(args) -> None:
    # Get executor
    if args.processor == "joblib":
        set_loky_pickler("pickle")
        executor = ExecutorJoblib(20)
    elif args.processor == "map":
        executor = ExecutorMap()
    executor_map = ExecutorMap()

    print("defined executor")

    # Get Dataset
    print("Loading datasets")
    dataset = MultiCrystalDataset.mcd_from_pandda_input_dir(p.Path(args.data_dirs))
    print("got datasets")

    print("\tBefore tuncation on res there are {} datasets".format(len(dataset.datasets)))
    dataset = truncate_dataset_on_resolution(dataset)
    print("\tAfter truncationt on res here are {} datasets".format(len(dataset.datasets)))

    # Select lowest res dataset
    min_res = get_min_res(dataset)


    # # Align structures
    # aligners =
    # alignments = mapdict(wrap_call,
    #                      alginers,
    #                      dataset.datasets.items(),
    #                      )
    #
    #
    # # Load xmaps
    # print("Getting exmpas")
    # xmaps = mapdict(Loader(min_res=None,
    #                        grid_params=grid_params,
    #                        ),
    #                 dataset.datasets,
    #                 executor,
    #                 )


    # Sample to reference frame
    cell = clipper_python.Cell(clipper_python.Cell_descr(100,100,100,np.pi/2, np.pi/2, np.pi/2))
    spacegroup = clipper_python.Spacegroup(clipper_python.Spgr_descr("1"))


    # Get reference model
    reference_model = list(dataset.datasets.items())[0][1].structure.structure
    io = PDBIO()
    io.set_structure(reference_model)
    io.save('out_before.pdb')
    atoms_list = np.array([atom.get_coord() for atom in reference_model.get_atoms()])
    print(atoms_list)
    mean_coords = np.mean(atoms_list, axis=0)
    print(mean_coords)
    # rotation = rotmat(Vector(0,1,0), Vector(1, 0, 0))
    rotation = np.eye(3)
    translation = np.array(mean_coords, 'f')
    for atom in reference_model.get_atoms():
        atom.transform(rotation, -translation)

    io = PDBIO()
    io.set_structure(reference_model)
    io.save('out.pdb')
    # exit()

    # Get model alignmetns
    aligners = {dtag: lambda: align(ref_structure=reference_model, mobile_structure=d.structure.structure)
                for dtag, d
                in dataset.datasets.items()
                }
    print(aligners)
    alignments = mapdict(wrap_call,
                         aligners,
                         executor_map,
                         )


    # align structures
    def align_structures(alignment, structure, output_path):
        alignment.apply(structure)
        io = PDBIO()
        io.set_structure(structure)
        print("Saving to {}".format(output_path))
        io.save(str(output_path))
        return structure

    structure_aligners = {dtag: lambda: align_structures(alignments[dtag],
                                                         d.structure.structure,
                                                         p.Path(args.out_dir) / dtag,
                                                         )
                          for dtag, d
                          in dataset.datasets.items()}
    print(structure_aligners)
    print("aligning and outputting structures")
    aligned_structures = mapdict(wrap_call,
                         structure_aligners,
                         executor_map,
                         )


    # structure_aligners = {dtag: lambda: alignment.apply(dataset.dataset[dtag].structure.structure)
    #                              for dtag, alignment
    #                              in alignments.items()
    #                              }
    # aligned_structures = mapdict(wrap_call,
    #                      structure_aligners,
    #                      executor_map,
    #                      )
    #
    # # ouput aligned structures
    # structure_outputters = {dtag: lambda: PDBIO().set_structure()}

    exit()






    # Convert xmaps to np
    print("Converting xmaps to np")
    static_structure = dataset.datasets[list(dataset.datasets.keys())[0]].structure.structure
    # print("Got static structure")
    aligners = {}
    for dtag, xmap in xmaps.items():
        aligners[dtag] = StructureAligner(static_structure,
                                          dataset.datasets[dtag].structure.structure,
                                          xmap,
                                          )

    xmaps_aligned = mapdict(wrap_call,
                            aligners,
                            executor,
                            )

    # Rescale
    rescaler = Rescaler()
    xmaps_np = mapdict(rescaler,
                       xmaps_aligned,
                       executor,
                       )
    postcondtion_scaling(xmaps_np)

    # Align xmaps
    print("aligning xmaps")

    xmaps_np = filter_on_grid(xmaps_np)


    # Embed the xmaps into a latent space
    print("Dimension reducing")
    xmap_embedding = embed_xmaps(xmaps_np.values())

    # Cluster the embedding
    print("clustering")
    clustered_xmaps = cluster_embedding(xmap_embedding)

    # Make dataframe with cluster and position
    print("getting cluster dataframe")
    cluster_df = get_cluster_df(xmaps_np,
                                xmap_embedding,
                                clustered_xmaps,
                                )
    cluster_df_summary(cluster_df)

    # Get clusters
    print("associating xmaps with clusters")
    map_clusters = get_map_clusters(cluster_df,
                                    xmaps_aligned,
                                    )

    #  Make mean maps
    executor_seriel = ExecutorMap()
    mean_maps_np = mapdict(lambda x: make_mean_map(x),
                           map_clusters,
                           executor_seriel,
                           )

    # Output the mean maps
    print("Outputting mean maps")
    template_map = xmaps[list(xmaps.keys())[0]]
    # print(template_map)
    # cell = dataset.datasets[list(dataset.datasets.keys())[0]].reflections.hkl_info.cell
    cell = clipper_python.Cell(clipper_python.Cell_descr(100, 100, 100, np.pi / 2, np.pi / 2, np.pi / 2))
    for cluster_num, mean_map_np_list in mean_maps_np.items():
        output_mean_map(template_map,
                        mean_map_np_list,
                        p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
                        )
        # output_mean_nxmap(mean_map_np_list,
        #                   cell,
        #                   p.Path(args.out_dir) / "{}.ccp4".format(cluster_num),
        #                   )

    # Ouptut the csv
    print("Outputting csv")
    output_labeled_embedding_csv(cluster_df,
                                 str(p.Path(args.out_dir) / "labelled_embedding.csv"),
                                 )

    # Output the graph
    # output_cluster_graph(cluster_df, str(out_dir / "output_graph.png"))

    return cluster_df


if __name__ == "__main__":
    # Parse Input
    args = get_args().parse_args()

    cluster_datasets(args)