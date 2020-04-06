import os
from pathlib import Path

import numpy as np

import luigi

# import clipper_python

import mdc3
from mdc3.types.files import PDBFile
from mdc3.types.structures import structure_biopython_from_pdb
from mdc3.functions.structures import translate_structure

from dataset_clustering.data_handling import copy_dataset
from dataset_clustering.align_maps import align_map_luigi
from dataset_clustering.cluster import cluster_datasets_luigi

import gemmi


def mark_done(path):
    f = open(str(path),
             "w",
             )
    f.write("done!")
    f.close()


class CopyDataset(luigi.Task):
    dtag = luigi.Parameter()
    dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        copy_dataset(self.dtag,
                     self.dataset_path,
                     self.output_path,
                     )
        mark_done(self.output_path / "done.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done.txt")


class AlignDataset(luigi.Task):
    dtag = luigi.Parameter()
    reference_dtag = luigi.Parameter()
    dataset_path = luigi.Parameter()
    reference_dataset_path = luigi.Parameter()
    output_path = luigi.Parameter()
    min_res = luigi.Parameter()

    def run(self):
        os.mkdir(str(self.output_path))
        align_map_luigi(self.dtag,
                        self.reference_dtag,
                        self.dataset_path,
                        self.reference_dataset_path,
                        self.output_path,
                        self.min_res,
                        )
        mark_done(self.output_path / "done.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done.txt")


class ClusterDatasets(luigi.Task):
    ccp4_map_paths = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        cluster_df = cluster_datasets_luigi(self.ccp4_map_paths,
                                            self.output_path,
                                            "map",
                                            20,
                                            )
        cluster_df.to_csv(str(self.output_path / "clustering.csv"))
        mark_done(self.output_path / "done.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done.txt")


class NormaliseStructure(luigi.Task):
    reference_pdb_path = luigi.Parameter()
    dtag = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        normalise_structure(self.reference_pdb_path,
                            self.dtag,
                            self.output_path,
                            )
        mark_done(self.output_path / "done_reference.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done_reference.txt")


def normalise_structure(reference_pdb_path, dtag, output_path):
    f = PDBFile(reference_pdb_path)
    structure = structure_biopython_from_pdb(f)

    box_origin = np.min(np.vstack([atom.coord
                                   for atom
                                   in structure.structure.get_atoms()
                                   ]
                                  ),
                        axis=0)
    print("\tBox origin is: {}".format(box_origin))

    translated_structure = translate_structure(structure, np.eye(3), -box_origin)

    box_origin = np.min(np.vstack([atom.coord
                                   for atom
                                   in translated_structure.structure.get_atoms()
                                   ]
                                  ),
                        axis=0)
    print(box_origin)

    translated_structure.output(output_path / "{}_normalised.pdb".format(dtag))


class AlignMapToReference(luigi.Task):
    dtag = luigi.Parameter()
    reference_dtag = luigi.Parameter()
    dataset_path = luigi.Parameter()
    reference_pdb_path = luigi.Parameter()
    output_path = luigi.Parameter()
    min_res = luigi.Parameter()
    structure_factors = luigi.Parameter()

    def run(self):
        try:
            os.mkdir(str(self.output_path))
        except Exception as e:
            print(e)

        align_map_to_reference(self.dtag,
                               self.reference_dtag,
                               self.dataset_path,
                               self.reference_pdb_path,
                               self.output_path,
                               self.min_res,
                               structure_factors=self.structure_factors,
                               )

        mark_done(self.output_path / "done.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done.txt")


def align_map_to_reference(dtag,
                           reference_dtag,
                           dataset_path,
                           reference_pdb_path,
                           output_path,
                           min_res,
                           structure_factors="FWT,PHWT",
                           ):
    # Load structures
    f_ref = PDBFile(reference_pdb_path)
    reference_structure = structure_biopython_from_pdb(f_ref)
    f_moving = PDBFile(dataset_path["pdb_path"])
    moving_structure = structure_biopython_from_pdb(f_moving)

    # Load xmap
    xmap = mdc3.types.real_space.xmap_from_path(dataset_path["mtz_path"],
                                                structure_factors,
                                                )

    # Get box limits from reference structure
    box_limits = np.max(np.vstack([atom.coord
                                   for atom
                                   in reference_structure.structure.get_atoms()
                                   ]
                                  ),
                        axis=0,
                        )
    print(box_limits)

    # Align and Get RTop to moving protein frame from alignment
    alignment_moving_to_ref = mdc3.functions.alignment.align(reference_structure.structure,
                                                             moving_structure.structure,
                                                             )
    alignment_ref_to_moving = mdc3.functions.alignment.align(moving_structure.structure,
                                                             reference_structure.structure,
                                                             )

    rotation = alignment_moving_to_ref.rotran[0]
    translation = alignment_moving_to_ref.rotran[1]
    alignment_moving_to_ref.apply(moving_structure.structure)
    print("translation: orthogonal to grid")
    print(translation)
    print("rotation")
    print(rotation)

    # Interpolate NX map in moving protein frame
    grid_params = [int(x) + 5 for x in box_limits]
    nxmap = mdc3.types.real_space.interpolate_uniform_grid(xmap,
                                                           translation,
                                                           np.transpose(rotation),
                                                           grid_params=grid_params,
                                                           )
    nxmap_data = nxmap.export_numpy()
    origin_nxmap = clipper_python.NXmap_float(clipper_python.Grid(grid_params[0],
                                                                  grid_params[1],
                                                                  grid_params[2],
                                                                  ),
                                              clipper_python.RTop_orth(clipper_python.Mat33_double(np.eye(3)),
                                                                       clipper_python.Vec3_double(0, 0, 0),
                                                                       )
                                              )
    origin_nxmap.import_numpy(clipper_python.Coord_grid(0, 0, 0),
                              nxmap_data,
                              )

    # Output to ccp4
    # cell = xmap.xmap.cell
    cell = clipper_python.Cell(clipper_python.Cell_descr(grid_params[0],
                                                         grid_params[1],
                                                         grid_params[2],
                                                         np.pi / 2,
                                                         np.pi / 2,
                                                         np.pi / 2,
                                                         )
                               )
    mdc3.types.real_space.output_nxmap(origin_nxmap,
                                       output_path / "{}_origin.ccp4".format(dtag),
                                       cell,
                                       )
    mdc3.types.real_space.output_nxmap(nxmap,
                                       output_path / "{}.ccp4".format(dtag),
                                       cell,
                                       )

    # Output aligned pdb
    moving_structure.output(output_path / "{}_aligned.pdb".format(dtag))


class MTZToCCP4(luigi.Task):

    dtag = luigi.Parameter()
    reference_dtag = luigi.Parameter()
    dataset_path = luigi.Parameter()
    reference_pdb_path = luigi.Parameter()
    output_path = luigi.Parameter()
    min_res = luigi.Parameter()
    structure_factors = luigi.Parameter()

    def run(self):
        try:
            os.mkdir(str(self.output_path))
        except Exception as e:
            print(e)

        self.mtz_to_ccp4(self.dtag,
                         self.reference_pdb_path,
                         self.dataset_path,
                         self.output_path,
                         self.min_res,
                         structure_factors=self.structure_factors,
                         )

        mark_done(self.output_path / "done.txt")

    def output(self):
        return luigi.LocalTarget(self.output_path / "done.txt")

    def mtz_to_ccp4(self,
                    dtag,
                    reference_pdb_path,
                    dataset_path,
                    output_path,
                    min_res,
                    structure_factors="FWT,PHWT",
                    ):

        # Load structures
        f_ref = PDBFile(reference_pdb_path)
        reference_structure = structure_biopython_from_pdb(f_ref)

        # Load xmap
        # xmap = mdc3.types.real_space.xmap_from_path(dataset_path["mtz_path"],
        #                                             structure_factors,
        #                                             )

        # Get box limits from reference structure
        box_limits_max = np.max(np.vstack([atom.coord
                                           for atom
                                           in reference_structure.structure.get_atoms()
                                           ]
                                          ),
                                axis=0,
                                )
        box_limits_min = np.min(np.vstack([atom.coord
                                           for atom
                                           in reference_structure.structure.get_atoms()
                                           ]
                                          ),
                                axis=0,
                                )

        # Interpolate NX map in moving protein frame
        grid_params = [int(x) + 4
                       for x
                       in (box_limits_max - box_limits_min)
                       ]

        # nxmap = mdc3.types.real_space.interpolate_uniform_grid(xmap,
        #                                                        box_limits_min - np.array([2, 2, 2]),
        #                                                        np.eye(3),
        #                                                        grid_params=grid_params,
        #                                                        )
        #
        # # Output to ccp4
        # cell = clipper_python.Cell(clipper_python.Cell_descr(grid_params[0],
        #                                                      grid_params[1],
        #                                                      grid_params[2],
        #                                                      np.pi / 2,
        #                                                      np.pi / 2,
        #                                                      np.pi / 2,
        #                                                      )
        #                            )
        #
        # mdc3.types.real_space.output_nxmap(nxmap,
        #                                    output_path / "{}.ccp4".format(dtag),
        #                                    cell,
        #                                    )
        mtz = gemmi.read_mtz_file(str(dataset_path["mtz_path"]))

        all_data = np.array(mtz, copy=False)
        mtz.set_data(all_data[mtz.make_d_array() >= 3.0])

        grid = mtz.transform_f_phi_to_map("FWT",
                                          "PHWT",
                                          sample_rate=3,
                                          )

        mp = gemmi.Ccp4Map()
        mp.grid = grid
        mp.update_ccp4_header(2, True)

        mp.write_ccp4_map(str(output_path / "{}.ccp4".format(dtag)))



        # arr = numpy.zeros([64, 64, 64], dtype=numpy.float32)
        # tr = gemmi.Transform()
        # tr.mat.fromlist(np.eye(3).tolist())
        # tr.vec.fromlist((box_limits_min - np.array([2, 2, 2])).lolist())
        # map.interpolate_values(arr,
        #                        tr,
        #                        )
        #
