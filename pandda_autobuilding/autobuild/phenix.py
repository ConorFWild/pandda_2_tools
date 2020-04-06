import os
import time
import functools
import subprocess
import shutil

from pathlib import Path

import pandas as pd

from autobuild.parallel import process
from autobuild.coarse import coarse_build
from autobuild.strip import strip_receptor_waters
from autobuild.events import get_events
from autobuild.result import Result

from autobuild.io import (parse_pandda_pdbs,
                          parse_pandda_event_maps,
                          parse_ligand_pdbs,
                          parse_pandda_event_table,
                          get_resolutions,
                          make_output_dirs,
                          output_results_table,
                          )


def ligandfit(out_dir_path,
              mtz,
              ligand,
              receptor,
              event_centroid,
              ):
    # REMOVE OLD LIGANDFIT RUNS

    if event_centroid is not None:
        command_string = "module load phenix; cd {out_dir_path}; phenix.ligandfit data={mtz} ligand={ligand} model={receptor} search_center=[{x},{y},{z}] search_dist=6"
        formatted_command = command_string.format(out_dir_path=out_dir_path,
                                                  mtz=mtz,
                                                  ligand=ligand,
                                                  receptor=receptor,
                                                  x=event_centroid[0],
                                                  y=event_centroid[1],
                                                  z=event_centroid[2],
                                                  )

    else:
        command_string = "module load phenix; cd {out_dir_path}; phenix.ligandfit data={mtz} ligand={ligand} model={receptor}"
        formatted_command = command_string.format(out_dir_path=out_dir_path,
                                                  mtz=mtz,
                                                  ligand=ligand,
                                                  receptor=receptor,
                                                  )

    print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    # print(stdout)
    # print(stderr)

    return out_dir_path / "LigandFit_run_1_"


def autobuild_event_phenix(protein_model_path: Path,
                           ligand_model_path: Path,
                           event_mtz_path: Path,
                           output_dir_path: Path,
                           event_centroid,
                           ):

    try:
        os.mkdir(str(output_dir_path))
    except Exception as e:
        print(e)

    phenix_tmp_out_dir = output_dir_path / "PDS"

    # phenix_out_dir = output_dir_path / "LigandFit_run_1_"

    ligand_fit_out_dir = output_dir_path / "LigandFit_run_1_"

    phenix_out_dir_regex = "LigandFit_run_*"

    phenix_output_regex = "ligand_fit_*.pdb"


    try:
        shutil.rmtree(str(phenix_tmp_out_dir),
                      ignore_errors=True,
                      )
    except Exception as e:
        print(e)

    ligand_fit_runs = output_dir_path.glob(phenix_out_dir_regex)
    for phenix_run_path in ligand_fit_runs:
        try:
            shutil.rmtree(str(phenix_run_path),
                          ignore_errors=True,
                          )
        except Exception as e:
            print(e)

    t_start = time.time()
    ligandfit(out_dir_path=output_dir_path,
              mtz=event_mtz_path,
              ligand=ligand_model_path,
              receptor=protein_model_path,
              event_centroid=event_centroid,
              )
    t_finish = time.time()

    built_ligand_paths = list(output_dir_path.glob("**/ligand_fit_*.pdb"))


    results: Result = Result(result_model_paths=built_ligand_paths,
                             success=(len(built_ligand_paths) > 0),
                             time=t_finish - t_start,
                             )

    return results
