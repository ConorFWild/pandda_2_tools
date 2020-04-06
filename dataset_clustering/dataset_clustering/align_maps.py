import os
import subprocess

import joblib


def wrap_call(obj):
    return obj()


class Aligner:
    def __init__(self, reference_pdb_path, reference_mtz_path, moving_pdb, moving_mtz_path, output_dir, reference_dtag,
                 dmin=3.0):
        self.reference_pdb_path = reference_pdb_path
        self.reference_mtz_path = reference_mtz_path
        self.moving_pdb = moving_pdb
        self.moving_mtz_path = moving_mtz_path
        self.output_dir = output_dir
        self.reference_dtag = reference_dtag
        self.dmin = dmin

    def __call__(self):
        align_map(self.reference_pdb_path,
                  self.reference_mtz_path,
                  self.moving_pdb,
                  self.moving_mtz_path,
                  self.output_dir,
                  self.reference_dtag,
                  self.dmin
                  )


def align_maps(reference_dataset_dtag, dataset_paths, output_dir, dmin=3.0, n_procs=20):

    reference_dtag = list(dataset_paths.keys())[0]

    reference_pdb_path = dataset_paths[reference_dtag]["pdb_path"]
    reference_mtz_path = dataset_paths[reference_dtag]["mtz_path"]

    aligners = {}
    for dtag in dataset_paths:
        if dtag == reference_dataset_dtag:
            dataset_output_dir = output_dir / dtag
            os.mkdir(str(dataset_output_dir))
            copy_file(dataset_paths[dtag]["pdb_path"],
                      dataset_output_dir / "{}.pdb".format(dtag),
                      )
            copy_file(dataset_paths[dtag]["mtz_path"],
                      dataset_output_dir / "{}.mtz".format(dtag),
                      )
            continue

        moving_pdb_path = dataset_paths[dtag]["pdb_path"]
        moving_mtz_path = dataset_paths[dtag]["mtz_path"]
        dataset_output_dir = output_dir / dtag
        os.mkdir(str(dataset_output_dir))
        aligners[dtag] = Aligner(reference_pdb_path,
                                 reference_mtz_path,
                                 moving_pdb_path,
                                 moving_mtz_path,
                                 dataset_output_dir,
                                 reference_dataset_dtag,
                                 dmin,
                                 )

    # print("\t\tSubmiting {} datasets for alignment".format(len(aligners)))
    joblib.Parallel(n_jobs=n_procs)(joblib.delayed(aligner)() for aligner in aligners.values())
    # for dtag, aligner in aligners.items():
    #     print("\t\t\tAligning dataset: {}".format(dtag))
    #     aligner()


def align_map_luigi(dtag,
                    reference_dtag,
                    dataset_path,
                    reference_dataset_path,
                    output_path,
                    min_res,
                    ):
    reference_pdb_path = reference_dataset_path["pdb_path"]
    reference_mtz_path = reference_dataset_path["mtz_path"]
    moving_pdb_path = dataset_path["pdb_path"]
    moving_mtz_path = dataset_path["mtz_path"]

    ouput_dir_path = output_path

    command = "module load phenix; cd {output_dir}; phenix.superpose_maps map_1={map_1} pdb_1={pdb_1} map_2={map_2} pdb_2={pdb_2} output_dir={output_dir} labels_1={labels_1} labels_2={labels_2} d_min_1={dmin} d_min_2={dmin}; rm {reference_dtag}*; cd -"
    formated_command = command.format(map_1=reference_mtz_path,
                                      pdb_1=reference_pdb_path,
                                      map_2=moving_mtz_path,
                                      pdb_2=moving_pdb_path,
                                      output_dir=ouput_dir_path,
                                      labels_1="DELFWT,PHDELWT",
                                      labels_2="DELFWT,PHDELWT",
                                      reference_dtag=reference_dtag,
                                      dmin=min_res,
                                      )

    # print("\t\tSubmiting alignment command: {}".format(formated_command))

    p = subprocess.Popen(formated_command,
                         shell=True,
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         )
    stdout, stderr = p.communicate()
    # print("\t\t{}".format(stdout))
    # print("\t\t{}".format(stderr))


def align_map(reference_pdb_path, reference_mtz_path, moving_pdb_path, moving_mtz_path, ouput_dir_path, reference_dtag, dmin=3.0):
    command = "cd {output_dir}; phenix.superpose_maps map_1={map_1} pdb_1={pdb_1} map_2={map_2} pdb_2={pdb_2} output_dir={output_dir} labels_1={labels_1} labels_2={labels_2} d_min_1={dmin} d_min_2={dmin}; rm {reference_dtag}*; cd -"
    formated_command = command.format(map_1=reference_mtz_path,
                                      pdb_1=reference_pdb_path,
                                      map_2=moving_mtz_path,
                                      pdb_2=moving_pdb_path,
                                      output_dir=ouput_dir_path,
                                      labels_1="FWT,PHWT",
                                      labels_2="FWT,PHWT",
                                      reference_dtag=reference_dtag,
                                      dmin=dmin,
                                      )

    # print("\t\tSubmiting alignment command: {}".format(formated_command))

    p = subprocess.Popen(formated_command,
                         shell=True,
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         )
    stdout, stderr = p.communicate()
    # print("\t\t{}".format(stdout))
    # print("\t\t{}".format(stderr))


def copy_file(old_file, new_file):
    command = "cp {old_file} {new_file}"
    formated_command = command.format(old_file=old_file,
                                      new_file=new_file,
                                      )
    p = subprocess.Popen(formated_command,
                         shell=True,
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         )
    p.communicate()
