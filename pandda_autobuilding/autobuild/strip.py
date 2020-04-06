import numpy as np
from biopandas.pdb import PandasPdb

def strip_receptor_waters(receptor_path, placed_ligand_path, output_path):

    ligand = PandasPdb().read_pdb(str(placed_ligand_path))
    receptor = PandasPdb().read_pdb(str(receptor_path))

    ligand_coords = ligand.df["HETATM"][["x_coord", "y_coord", "z_coord"]]

    ligand_com = np.array(ligand_coords.mean(axis=0))

    receptor_hetatms = np.array(receptor.df["HETATM"][["x_coord", "y_coord", "z_coord"]])

    # print(receptor.df["HETATM"])

    distances_from_event = np.sqrt(np.sum(np.square(receptor_hetatms - ligand_com), axis=1))

    # print(distances_from_event)

    receptor_df_copy = receptor.df["HETATM"].copy()
    receptor_df_copy["distances"] = distances_from_event

    receptor.df["HETATM"] = receptor_df_copy[receptor_df_copy["distances"] > 10.0]

    # TODO: FIX HACK
    receptor.df["ATOM"][receptor.df["ATOM"]["b_factor"] > 99.0]["b_factor"] = 99.0
    receptor.df["HETATM"][receptor.df["HETATM"]["b_factor"] > 99.0]["b_factor"] = 99.0
    receptor.df["ATOM"]["b_factor"] = 99.0
    receptor.df["HETATM"]["b_factor"] = 99.0

    # print(receptor.df["ATOM"]["b_factor"])

    # print(receptor.df["HETATM"])


    receptor.to_pdb(path=output_path,
                  records=None,
                  gz=False,
                  append_newline=True,
                  )

    return output_path