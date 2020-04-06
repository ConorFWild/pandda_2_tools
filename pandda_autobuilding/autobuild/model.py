from pathlib import Path

import numpy as np
import pandas as pd

from biopandas.pdb import PandasPdb


class AbstractModel:

    def get_coords_array(self):
        raise NotImplementedError()


class BioPandasModel(AbstractModel):

    def __init__(self,
                 model_path: Path,
                 ):
        self.model = PandasPdb().read_pdb(str(model_path))

    def get_coords_array(self, residue_name="LIG"):
        df = self.model.df["HETATM"]
        coords_df: pd.DataFrame = df[df["residue_name"]==residue_name][["x_coord", "y_coord", "z_coord"]]
        coords_array = np.array(coords_df)

        return coords_array


def get_rmsd(model_1: AbstractModel, model_2: AbstractModel, residue_name="LIG"):
    # print(model_1.model.df["HETATM"])
    # print(model_2.model.df["HETATM"])

    model_1_coords_array = model_1.get_coords_array()
    model_2_coords_array = model_2.get_coords_array()

    distances = model_1_coords_array - model_2_coords_array
    distances_squared = np.square(distances)
    sum_of_squares = np.sum(distances_squared, axis=1)
    l2_distances = np.sqrt(sum_of_squares)
    mean_l2_distances = np.mean(l2_distances)

    return mean_l2_distances

def get_rmsd_dfs(model_1: pd.DataFrame, model_2: pd.DataFrame, residue_name="LIG"):
    # print(model_1.model.df["HETATM"])
    # print(model_2.model.df["HETATM"])

    model_1_coords_array = np.array(model_1)
    model_2_coords_array = np.array(model_2)

    distances = model_1_coords_array - model_2_coords_array
    distances_squared = np.square(distances)
    sum_of_squares = np.sum(distances_squared, axis=1)
    l2_distances = np.sqrt(sum_of_squares)
    mean_l2_distances = np.mean(l2_distances)

    return mean_l2_distances

def is_comparable(model_1: AbstractModel, model_2: AbstractModel):

    model_1_coords_array = model_1.get_coords_array()
    model_2_coords_array = model_2.get_coords_array()

    if len(model_1_coords_array) != len(model_2_coords_array):
        # raise Exception("Model 1 has length {}, but model 2 has length {}".format(len(model_1_coords_array),
        #                                                                           len(model_2_coords_array))
        #                 )
        return False
    else:
        return True

