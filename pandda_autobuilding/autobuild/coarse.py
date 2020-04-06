import numpy as np
from biopandas.pdb import PandasPdb


def coarse_build(ligand_model_path,
                 event,
                 output_path,
                 only_place=True,
                 ):
    # print(ligand_model_path)
    # print(event)
    # print(output_path)

    ligand = PandasPdb().read_pdb(str(ligand_model_path))

    ligand_coords = ligand.df["HETATM"][["x_coord", "y_coord", "z_coord"]]

    # print(ligand_coords)

    # print(ligand_coords)


    ligand_com = np.array(ligand_coords.mean(axis=0))

    # print(event)
    event_com = np.array([event.x, event.y, event.z])

    translation = event_com - ligand_com
    # print(translation)

    ligand.df["HETATM"]["x_coord"] = ligand.df["HETATM"]["x_coord"] + translation[0]
    ligand.df["HETATM"]["y_coord"] = ligand.df["HETATM"]["y_coord"] + translation[1]
    ligand.df["HETATM"]["z_coord"] = ligand.df["HETATM"]["z_coord"] + translation[2]

    # print(ligand.df["HETATM"][["x_coord", "y_coord", "z_coord"]])

    # print(ligand.df["HETATM"][["x_coord", "y_coord", "z_coord"]] - ligand_coords)


    ligand.to_pdb(path=output_path,
                  records=None,
                  gz=False,
                  append_newline=True,
                  )

    return output_path