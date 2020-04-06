from typing import NamedTuple, Dict
import logging
import pprint

logging.basicConfig(filename='luigi.log', level=logging.INFO)

import argparse
from pathlib import Path
from functools import partial
import os
import shutil

import numpy as np
import pandas as pd
import joblib
from joblib.externals.loky import set_loky_pickler

from Bio import PDB

import luigi

luigi.configuration.get_config().set('logging', 'log_level', 'INFO')

import clipper_python

import dataset_clustering
# from dataset_clustering.cluster import cluster_datasets_rsa as cluster_datasets
from dataset_clustering.cluster import cluster_datasets_rsa as cluster_datasets
from dataset_clustering.data_handling import copy_datasets
from dataset_clustering.cluster import cluster_datasets_luigi

import mdc3
from mdc3.functions.functional import (mapdict,
                                       wrap_call,
                                       )
from mdc3.functions.alignment import align
from mdc3.types.real_space import MCDXMap
from mdc3.types.datasets import (parse_pandda_input,
                                 parse_pandda_input_for_regex,
                                 )
from mdc3.functions.utils import res_from_mtz_file

from XChemDB.xchem_db import XChemDB

from dataset_clustering.luigi_lib import (CopyDataset,
                                          AlignDataset,
                                          ClusterDatasets,
                                          NormaliseStructure,
                                          normalise_structure,
                                          )
from dataset_clustering.clustering import (embed_xmaps,
                                           cluster_embedding,
                                           )
from dataset_clustering.partition_residues import partition_residues
from dataset_clustering.sample_map import Sampler
from dataset_clustering.graph import graph


class ClusterXCDBArgs(NamedTuple):
    root_path: Path
    out_dir: Path
    n_procs: int
    clean_run: bool
    # xcdb_path: Path


class ClusterArgs(NamedTuple):
    in_dir: Path
    out_dir: Path
    n_procs: int
    mtz_regex: str
    pdb_regex: str
    structure_factors: str


def get_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-i", "--in_dir",
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

    return parser


class ExecutorMap:
    def __init__(self):
        pass

    def __call__(self, func, values):
        results = map(func,
                      values,
                      )
        return results


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


class AlignAtoms:
    def __init__(self, ref_atoms, alt_atoms):
        self.ref_atoms = ref_atoms
        self.alt_atoms = alt_atoms

    def __call__(self):
        super_imposer = PDB.Superimposer()
        super_imposer.set_atoms(self.ref_atoms,
                                self.alt_atoms,
                                )

        return super_imposer


def cat_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs,
                   ignore_index=True,
                   )
    return df


def make_residue_cluster_df(res_id,
                            sample_clusters,
                            sample_embedding,
                            ):
    records = []

    sample_coords = {dtag: (sample_embedding[i, 0],
                            sample_embedding[i, 1],
                            )
                     for i, dtag
                     in enumerate(sample_clusters)
                     }

    for dtag, sample_cluster in sample_clusters.items():
        records.append({"res_id": res_id,
                        "dtag": dtag,
                        "sample_cluster": sample_cluster,
                        "x": sample_coords[dtag][0],
                        "y": sample_coords[dtag][1],
                        }
                       )

    df = pd.DataFrame(records)

    print(df)

    return df


def parse_args():
    cmd_args = get_args().parse_args()

    args = ClusterArgs(in_dir=Path(cmd_args.in_dir),
                       out_dir=Path(cmd_args.out_dir),
                       n_procs=int(cmd_args.n_procs),
                       mtz_regex=str(cmd_args.mtz_regex),
                       pdb_regex=str(cmd_args.pdb_regex),
                       structure_factors=str(cmd_args.structure_factors),
                       )

    return args


def get_datasets(inital_model_path, mtz_regex, pdb_regex, structure_factors):
    datasets = mdc3.types.datasets.MultiCrystalDataset.mcd_from_pandda_input_dir(inital_model_path,
                                                                                 mtz_regex=mtz_regex,
                                                                                 pdb_regex=pdb_regex,
                                                                                 structure_factors=structure_factors,
                                                                                 )
    return datasets


def get_reference(datasets):
    item = list(datasets.items())[0]
    return item[0], item[1]


def get_min_res(datasets):
    return 3.5


def dispatch(func_dict
             ):
    results = {}

    for i, func in func_dict.items():
        results[i] = func()

    return results


def load_xmap(dataset, min_res):
    xmap = Loader(dataset, min_res=min_res)()
    return xmap


def get_residues(structure,
                 chain_id,
                 res_id,
                 window,
                 ):
    residue_num = res_id[1]
    model = structure[0]
    chain = model[chain_id]

    residues = []
    try:
        for i in range((2 * int(window)) + 1):
            res = chain[(res_id[0],
                         (int(residue_num) + i) - window,
                         res_id[2],
                         )
            ]
            residues.append(res)
        # print("Aligning on: {}".format(residues))

    except:
        print("\tAligning on single atom!")
        residues.append(chain[res_id])

    return residues


def get_atoms(residues, name):
    atoms = []
    for residue in residues:
        atom = residue[name]
        atoms.append(atom)

    return atoms


def align_residues(structure,
                   reference_structure,
                   chain_id,
                   residue_id,
                   window=1,
                   ):
    dataset_residues = get_residues(structure,
                                    chain_id,
                                    residue_id,
                                    window,
                                    )
    # print("dataset_residues: {}".format(dataset_residues))

    reference_residues = get_residues(reference_structure,
                                      chain_id,
                                      residue_id,
                                      window,
                                      )
    # print("reference_residues: {}".format(reference_residues))

    atoms = get_atoms(dataset_residues, "CA")
    # print("atoms: {}".format(atoms))

    reference_residue_atoms = get_atoms(reference_residues, "CA")
    # print("reference_residue_atoms: {}".format(reference_residue_atoms))

    aligner = AlignAtoms(reference_residue_atoms,
                         atoms,
                         )
    alignment = aligner()

    return alignment


def get_coords(residue, name):
    return residue[name]


def sample_around_residue(xmap,
                          alignment,
                          residue,
                          grid_params=np.array([10, 10, 10],
                                               dtype=int,
                                               ),
                          ):
    reference_residue_com = get_coords(residue,
                                       "CA",
                                       )
    reference_residue_com_np = reference_residue_com.get_coord()

    sampler = Sampler(xmap,
                      alignment,
                      reference_residue_com_np,
                      grid_params=grid_params,
                      offset=grid_params / 2,
                      )

    return sampler()


def embed_samples(samples):
    embedding = embed_xmaps(samples.values())
    return embedding


def cluster(sample_embedding):
    clustered_samples = cluster_embedding(sample_embedding)
    return clustered_samples


def get_cluster_table(chain_id,
                      res_id,
                      sample_clusters,
                      sample_embedding,
                      ):
    records = []

    sample_coords = {dtag: (sample_embedding[i, 0],
                            sample_embedding[i, 1],
                            )
                     for i, dtag
                     in enumerate(sample_clusters)
                     }

    for dtag, sample_cluster in sample_clusters.items():
        records.append({"chain_id": chain_id,
                        "res_id": res_id,
                        "dtag": dtag,
                        "sample_cluster": sample_cluster,
                        "x": sample_coords[dtag][0],
                        "y": sample_coords[dtag][1],
                        }
                       )

    df = pd.DataFrame(records)

    # print(df)
    return df


def output_table(table,
                 out_path,
                 ):
    table.to_csv(str(out_path))


def output_graph(table,
                 out_path,
                 ):
    graph(table,
          out_path,
          x="x",
          y="y",
          id="dtag",
          col="sample_cluster",
          )


def iter_residues(dataset):
    structure = dataset.structure

    for chain in structure.get_chains():
        for residue in chain.get_residues():
            yield chain.get_id(), residue.get_id(), residue


if __name__ == "__main__":

    args = parse_args()
    print("args: {}".format(args))

    datasets = get_datasets(args.in_dir,
                            mtz_regex=args.mtz_regex,
                            pdb_regex=args.pdb_regex,
                            structure_factors=args.structure_factors,
                            )

    print("datasets: {}".format(datasets))

    reference_dtag, reference_dataset = get_reference(datasets.datasets)
    print("reference_dataset: {}".format(reference_dataset))

    min_res = get_min_res(datasets)
    print("min_res: {}".format(min_res))

    xmaps = dispatch({dtag: partial(load_xmap,
                                    dataset,
                                    min_res=min_res,
                                    )
                      for dtag, dataset
                      in datasets.datasets.items()
                      }
                     )

    for chain_id, res_id, residue in iter_residues(reference_dataset.structure):
        print("\tRes id: {}".format(res_id))

        alignments = dispatch({dtag: partial(align_residues,
                                             dataset.structure.structure,
                                             reference_dataset.structure.structure,
                                             chain_id,
                                             res_id,
                                             )
                               for dtag, dataset
                               in datasets.datasets.items()}
                              )
        # print("\talignments: {}".format(alignments))

        samples = dispatch({dtag: partial(sample_around_residue,
                                          xmap,
                                          alignments[dtag],
                                          residue,
                                          )
                            for dtag, xmap
                            in xmaps.items()
                            }
                           )
        # print("\tsamples: {}".format(samples))

        embedding = embed_samples(samples)
        # print("\tembedding: {}".format(embedding))

        clusters = cluster(embedding)
        # print("\tclusters: {}".format(clusters))

        sample_clusters = {dtag: clusters.labels_[i]
                           for i, dtag
                           in enumerate(samples)
                           }

        cluster_table = get_cluster_table(chain_id,
                                          res_id,
                                          sample_clusters,
                                          embedding,
                                          )
        # print("\tcluster_table: {}".format(cluster_table))
        print("\tClusters: {}".format(cluster_table["sample_cluster"].unique()))

        output_table(cluster_table,
                     args.out_dir / "{chain}_{residue}.csv".format(chain=chain_id,
                                                                   residue=res_id,
                                                                   ),
                     )

        output_graph(cluster_table,
                     args.out_dir / "{chain}_{residue}.html".format(chain=chain_id,
                                                                   residue=res_id,
                                                                   ),
                     )
        # except Exception as e:
        #     print("\tencountered exception: {}".format(e))
    #
    # cmd_args = get_args().parse_args()
    #
    # args = ClusterXCDBArgs(in_dir=Path(cmd_args.in_dir),
    #                        out_dir=Path(cmd_args.out_dir),
    #                        n_procs=int(cmd_args.n_procs),
    #                        clean_run=bool(cmd_args.clean_run),
    #                        )
    #
    # # Get executor
    # executor = ExecutorMap()
    #
    # # make output directory
    # try:
    #     os.mkdir(args.out_dir / system.system)
    # except:
    #     print("Already amde system dir")
    #
    # # Load datasets:  Dict[dtag, Dataset]
    # mcd = mdc3.types.datasets.MultiCrystalDataset.mcd_from_pandda_input_dir(Path(system.initial_model_path))
    #
    # # Partition residues:  Dict[res_num, Dict[dtag, Dict[dtag, Atom["CA"]]]]
    # partitioning = partition_residues({dtag: d.structure.structure
    #                                    for dtag, d
    #                                    in mcd.datasets.items()
    #                                    },
    #                                   length=3,
    #                                   )
    #
    # # Get xmaps
    # loaders = {dtag: Loader(d,
    #                         min_res=3.6,
    #                         )
    #            for dtag, d
    #            in mcd.datasets.items()
    #            }
    # xmaps = mapdict(wrap_call,
    #                 loaders,
    #                 executor,
    #                 )
    #
    # # Iterate over Dict[dtag, Dict[Resnum, Atom]]
    # for res_tuple, residues_dict in partitioning.items():
    #     res_id = res_tuple
    #     print("Resresidue_clustering.py id")
    #
    #     # Get reference residue
    #     reference_residue_dtag = list(residues_dict.keys())[0]
    #     reference_residue_atoms = residues_dict[reference_residue_dtag]
    #     reference_residue_com = np.mean(np.vstack([atom.coord
    #                                                for atom
    #                                                in reference_residue_atoms.values()]),
    #                                     axis=0,
    #                                     )
    #
    #     # Align triple:  Dict[dtag, Alignment]
    #     residue_aligners = {dtag: AlignAtoms(list(reference_residue_atoms.values()),
    #                                          list(atoms.values()),
    #                                          )
    #                         for dtag, atoms
    #                         in residues_dict.items()
    #                         }
    #     alignments = mapdict(wrap_call,
    #                          residue_aligners,
    #                          executor,
    #                          )
    #
    #     # Sample:  Dict[dtag, np.array]
    #     grid_params = np.array([20, 20, 20],
    #                            dtype=int,
    #                            )
    #     map_samplers = {dtag: Sampler(xmaps[dtag],
    #                                   alignments[dtag],
    #                                   reference_residue_com,
    #                                   grid_params=grid_params,
    #                                   offset=grid_params / 2,
    #                                   )
    #                     for dtag
    #                     in alignments
    #                     }
    #     samples = mapdict(wrap_call,
    #                       map_samplers,
    #                       executor,
    #                       )
    #
    #     print("Dimension reducing")
    #     sample_embedding = embed_xmaps(samples.values())
    #
    #     # Cluster the embedding
    #     print("clustering")
    #     clustered_samples = cluster_embedding(sample_embedding)
    #
    #     sample_clusters = {dtag: clustered_samples.labels_[i]
    #                        for i, dtag
    #                        in enumerate(samples)
    #                        }
    #
    #     # record:  pd.DataFrame
    #     print(sample_clusters)
    #     residue_clusters[res_id] = make_residue_cluster_df(res_id,
    #                                                        sample_clusters,
    #                                                        sample_embedding,
    #                                                        )
    #
    #     print("\tOutputting graph to {}".format(str(args.out_dir / "{}_{}_{}.html".format(res_id[0],
    #                                                                                       res_id[1],
    #                                                                                       res_id[2],
    #                                                                                       )
    #                                                 )
    #                                             )
    #           )
    #     graph(residue_clusters[res_id],
    #           str(args.out_dir / "{}_{}_{}.html".format(res_id[0],
    #                                                     res_id[1],
    #                                                     res_id[2],
    #                                                     )
    #               ),
    #           x="x",
    #           y="y",
    #           id="dtag",
    #           col="sample_cluster",
    #           )
