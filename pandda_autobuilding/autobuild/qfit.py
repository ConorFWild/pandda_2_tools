import time
import functools
import subprocess

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


def autobuild_qfit(pandda_dir_path, output_dir):
    event_table = parse_pandda_event_table(pandda_dir_path)
    print(event_table)

    events = get_events(event_table)
    print(events)

    input_pdbs = parse_pandda_pdbs(pandda_dir_path, events)
    print(input_pdbs)

    event_maps = parse_pandda_event_maps(pandda_dir_path, events)
    print(event_maps)

    ligand_pdbs = parse_ligand_pdbs(pandda_dir_path, events)
    print(ligand_pdbs)

    resolutions = get_resolutions(events)

    output_dirs = make_output_dirs(output_dir, events)

    placed_ligand_paths = process({event_id: functools.partial(coarse_build,
                                                               ligand_model_path=ligand_pdbs[event_id],
                                                               event=event,
                                                               output_path=output_dir / "{}_{}".format(event.dtag,
                                                                                                       event.event_idx) / "ligand.pdb",
                                                               )
                                   for event_id, event
                                   in events.items()
                                   }
                                  )

    stripped_receptor_paths = process({event_id: functools.partial(strip_receptor_waters,
                                                                   receptor_path=input_pdbs[event_id],
                                                                   placed_ligand_path=placed_ligand_paths[event_id],
                                                                   output_path=output_dir / "{}_{}".format(event.dtag,
                                                                                                           event.event_idx) / "stripped_receptor.pdb",
                                                                   )
                                       for event_id, event
                                       in events.items()
                                       }
                                      )

    qfit_paths = process({event_id: functools.partial(qfit,
                                                      protein_model_path=stripped_receptor_paths[event_id],
                                                      ligand_model_path=placed_ligand_paths[event_id],
                                                      event_map_path=event_maps[event_id],
                                                      resolution=resolutions[event_id],
                                                      output_dir_path=output_dirs[event_id],
                                                      )
                          for event_id, event
                          in events.items()
                          }
                         )

    results_table = process({idx: functools.partial(parse_qfit_results,
                                                    qfit_paths[idx],
                                                    )
                             for idx, event
                             in events.items()
                             }
                            )

    output_results_table(results_table)


def qfit(protein_model_path,
         ligand_model_path,
         event_map_path,
         resolution,
         output_dir_path,
         step=1,
         dof=1,
         ):
    command_string = "/dls/science/groups/i04-1/conor_dev/ccp4/base/bin/qfit_ligand {event_map} {resolution} {ligand_model_path} -r {protein_model_path} -d {output_dir_path} -s {step} -b {dof}"
    formatted_command = command_string.format(event_map=event_map_path,
                                              resolution=resolution,
                                              ligand_model_path=ligand_model_path,
                                              protein_model_path=protein_model_path,
                                              output_dir_path=output_dir_path,
                                              step=step,
                                              dof=dof,
                                              )

    print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return output_dir_path


def parse_qfit_results(qfit_results_path):
    pass


def autobuild_event_qfit(protein_model_path: Path,
                         ligand_model_path: Path,
                         event_map_path: Path,
                         resolution: float,
                         output_dir_path: Path,
                         ):

    t_start = time.time()
    qfit(protein_model_path,
         ligand_model_path,
         event_map_path,
         resolution,
         output_dir_path,
         )
    t_finish = time.time()
    # results: pd.DataFrame = parse_qfit_results(output_dir_path)

    results: Result = Result(result_model_paths=list(output_dir_path.glob("conformer_*.pdb")),
                             success=(output_dir_path / "conformer_1.pdb").is_file(),
                             time=t_finish - t_start,
                             )

    return results
