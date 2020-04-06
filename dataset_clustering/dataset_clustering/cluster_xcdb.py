from typing import NamedTuple, Dict
import argparse
from pathlib import Path
import os
import shutil

import pandas as pd



# from cluster import cluster_datasets_tm as cluster_datasets
from cluster import cluster_datasets_rsa as cluster_datasets


from XChemDB.xchem_db import XChemDB


class ClusterXCDBArgs(NamedTuple):
    root_path: Path
    out_dir: Path
    # xcdb_path: Path


class ClusterArgs(NamedTuple):
    data_dirs: Path
    out_dir: Path
    processor: str


def get_args():
    parser = argparse.ArgumentParser()

    # IO
    # parser.add_argument("-c", "--config_path",
    #                     type=str,
    #                     default="config.ini",
    #                     help="Path to the config file with uncommon options"
    #                     )
    # parser.add_argument("-d", "--data_dirs",
    #                     type=str,
    #                     help="The directory dir such that dir//<dataset_names>//<pdbs and mtzs>",
    #                     required=True
    #                     )
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
    # parser.add_argument("-n", "--num_datasets",
    #                     type=str,
    #                     help="The number of datasets to use in a random sample",
    #                     required=True
    #                     )
    # parser.add_argument("-p", "--processor",
    #                     type=str,
    #                     help="The processing method to use",
    #                     default="map",
    #                     )

    return parser


def cat_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(dfs,
                   ignore_index=True,
                   )
    return df


def get_cluster_dfs(args: ClusterXCDBArgs) -> pd.DataFrame:

    xchem_db = XChemDB.from_file_tree(args.root_path)

    cluster_dfs: Dict[str, pd.DataFrame] = {}

    for i, system in enumerate(xchem_db.database.itertuples()):

        # try:
        if system.system != "MUREECA":
        # if system.system != "XX02KALRNA":
            print("Passing on large system")
            continue
            # raise Exception("Passing on large system")

        print("System intial models: {}".format(system.initial_model_path))
        # print(system.system)
        # print(args.out_dir / system.system)
        system_out_dir = args.out_dir / system.system
        # print(system_out_dir.resolve())
        # raise Exception("")


        cluster_args = ClusterArgs(data_dirs=system.initial_model_path,
                           out_dir=system_out_dir,
                           processor="joblib",
                           )

        try:
            shutil.rmtree(system_out_dir.resolve())
        except:
            pass

        os.mkdir(system_out_dir.resolve())

        cluster_dfs[str(system.system)] = cluster_datasets(cluster_args)

        print("######################{}######################".format(i))
        # exit()

        # if i >2:
        #     break
        # except Exception as e:
        print("###########################################################")
        print("WARNING: Failed on system: {}".format(system.system))
        print("############################################################")
        print(e)
    cluster_dfs_df = cat_dfs(cluster_dfs)

    return cluster_dfs_df






if __name__ == "__main__":

    cmd_args = get_args().parse_args()

    args = ClusterXCDBArgs(root_path=Path(cmd_args.root_path),
                           out_dir=Path(cmd_args.out_dir),
                           )

    cluster_dfs_df = get_cluster_dfs(args)

    cluster_dfs_df.to_csv(str(Path(cmd_args.out_dir) / "clusters.csv"))


