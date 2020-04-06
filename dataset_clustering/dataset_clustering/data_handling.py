import os
import subprocess
import pandas as pd


def get_cluster_df(dataset, xmap_embedding, clustered_xmaps):
    records = []


    for i, key in enumerate(dataset):
        record = {"dtag": key,
                  "x": xmap_embedding[i, 0],
                  "y": xmap_embedding[i, 1],
                  "cluster": clustered_xmaps[i]
                  }
        records.append(record)

    df = pd.DataFrame(records)

    return df


def output_labeled_embedding_csv(cluster_df, path):
    cluster_df.to_csv(path)


def get_map_clusters(cluster_df, xmaps):
    map_clusters = {}
    for cluster_num in cluster_df["cluster"].unique():
        cluster_dtags = cluster_df[cluster_df["cluster"] == cluster_num]["dtag"].values
        map_clusters[cluster_num] = [xmap for dtag, xmap in xmaps.items() if (dtag in cluster_dtags)]

    return map_clusters


def copy_file(file_path, target_path):

    command = "cp {old_path} {new_path}"
    formatted_comand = command.format(old_path=file_path,
                                      new_path=target_path)
    p = subprocess.Popen(formatted_comand,
                         shell=True,
                         )
    p.communicate()


def copy_datasets(datasets, new_dir):
    for dtag in datasets:
        dataset_output_dir = new_dir / dtag
        os.mkdir(str(dataset_output_dir))
        new_pdb_path = dataset_output_dir / "{}.pdb".format(dtag)
        new_mtz_path = dataset_output_dir / "{}.mtz".format(dtag)
        dataset_paths = datasets[dtag]
        copy_file(dataset_paths["pdb_path"],
                  new_pdb_path,
                  )
        copy_file(dataset_paths["mtz_path"],
                  new_mtz_path,
                  )


def copy_dataset(dtag,
                 dataset_paths,
                 dir,
                 ):

    dataset_output_dir = dir
    os.mkdir(str(dataset_output_dir))
    new_pdb_path = dataset_output_dir / "{}.pdb".format(dtag)
    new_mtz_path = dataset_output_dir / "{}.mtz".format(dtag)

    copy_file(dataset_paths["pdb_path"],
              new_pdb_path,
              )
    copy_file(dataset_paths["mtz_path"],
              new_mtz_path,
              )
