from typing import NamedTuple, Dict, List
import os
import logging
import subprocess

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pandda_3.types.data_types import Event

from autobuild.cmd import (autobuild_from_cmd,
                           )
from autobuild.qfit import autobuild_event_qfit
from autobuild.phenix import autobuild_event_phenix
from autobuild.coarse import coarse_build
from autobuild.strip import strip_receptor_waters
from autobuild.model import (BioPandasModel,
                             get_rmsd,
                             is_comparable,
                             get_rmsd_dfs,
                             )
from autobuild.result import Result


def parse_args():
    parser = argparse.ArgumentParser()
    # IO

    parser.add_argument("-i", "--events_csv_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-o", "--output_dir_path",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    args = parser.parse_args()

    return args


class Config(NamedTuple):
    events_csv_path: Path
    output_dir: Path


def get_config(args):
    config: Config = Config(events_csv_path=Path(args.events_csv_path),
                            output_dir=Path(args.output_dir_path),
                            )

    return config


def get_table(event_csv_path: Path):
    return pd.read_csv(str(event_csv_path))


def get_events(events_table):
    events: List[Event] = []

    for idx, event_record in events_table.iterrows():
        event: Event = Event(dtag=str(event_record["dtag"]),
                             event_idx=int(event_record["event_idx"]),
                             occupancy=event_record["occupancy"],
                             analysed_resolution=event_record["analysed_resolution"],
                             high_resolution=event_record["high_resolution"],
                             interesting=event_record["interesting"],
                             ligand_placed=event_record["ligand_placed"],
                             ligand_confidence=event_record["ligand_confidence"],
                             viewed=event_record["viewed"],
                             initial_model_path=Path(event_record["initial_model_path"]),
                             data_path=Path(event_record["data_path"]),
                             final_model_path=Path(event_record["final_model_path"]),
                             event_map_path=Path(event_record["event_map_path"]),
                             actually_built=event_record["actually_built"],
                             x=event_record["x"],
                             y=event_record["y"],
                             z=event_record["z"],
                             )

        events.append(event)

    return events


def make_output_dir(output_dir: Path, events):
    try:
        os.mkdir(str(output_dir))
    except:
        print("\tAlready made main output!")

    new_events = []

    for event in events:

        if not is_event_built(event):
            continue

        event_dir_path: Path = output_dir / "{}_{}".format(event.dtag, event.event_idx)

        already_built = []
        try:
            os.mkdir(str(event_dir_path))

        except:
            already_built.append("{}_{}".format(event.dtag, event.event_idx))

        new_events.append(event)

    return new_events


def is_event_built(event: Event):
    if event.actually_built:
        return True
    else:
        return False

    # event_model_path: Path = event.final_model_path
    #
    # if event_model_path.is_file():
    #     model = BioPandasModel(event.final_model_path)
    #     df = model.model.df["HETATM"]
    #
    #     if len(df[df["residue_name"] == "LIG"]) != 0:
    #         return True
    #     else:
    #         return False
    # else:
    #     return False


def get_protein_model_path(event: Event):
    return event.initial_model_path


def get_ligand_model_path(event: Event):
    event_dir: Path = event.initial_model_path.parent

    ligands = list((event_dir / "ligand_files").glob("*.pdb"))
    ligand_strings = [str(ligand_path) for ligand_path in ligands if ligand_path.name != "tmp.pdb"]

    ligand_pdb_path: Path = Path(min(ligand_strings,
                                     key=len,
                                     )
                                 )
    return ligand_pdb_path


def get_ligand_cif_path(event: Event):
    event_dir: Path = event.initial_model_path.parent

    ligand_cifs = list((event_dir / "ligand_files").glob("*.cif"))
    ligand_cif_strings = [str(ligand_cif_path)
                          for ligand_cif_path
                          in ligand_cifs
                          if ligand_cif_path.name != "tmp.cif"
                          ]

    ligand_cif_path: Path = Path(min(ligand_cif_strings,
                                     key=len,
                                     )
                                 )
    return ligand_cif_path


def get_event_map_path(event: Event):
    return event.event_map_path


def event_map_to_mtz(event_map_path: Path,
                     output_path,
                     resolution,
                     col_f="FWT",
                     col_ph="PHWT",
                     gemmi_path: Path = "/dls/science/groups/i04-1/conor_dev/gemmi/gemmi",
                     ):
    command = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl; {gemmi_path} map2sf {event_map_path} {output_path} {col_f} {col_ph} --dmin={resolution}"
    formatted_command = command.format(gemmi_path=gemmi_path,
                                       event_map_path=event_map_path,
                                       output_path=output_path,
                                       col_f=col_f,
                                       col_ph=col_ph,
                                       resolution=resolution,
                                       )
    print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return output_path


def get_results(results: Dict[str, Result],
                true_model_path: Path,
                ):
    true_model = BioPandasModel(true_model_path)

    method_results = []

    for build_name, result in results.items():
        record = {}
        record["method"] = build_name
        record["num_candidates"] = len(result.result_model_paths)

        candidate_model_results = []

        for candidate_model_path in result.result_model_paths:
            candidate_model_record = {}

            result_model: BioPandasModel = BioPandasModel(candidate_model_path)

            # Loop over potential chain

            rmsds = []
            for chain_id in true_model.model.df["HETATM"]["chain_id"].unique():
                # if not is_comparable(true_model, result_model):
                #     print("True model and built model of different lengths! Cannot compare!")
                #     continue
                #

                true_model_df = true_model.model.df["HETATM"][true_model.model.df["HETATM"]["chain_id"] == chain_id][
                    ["x_coord", "y_coord", "z_coord"]]
                result_model_df = \
                    result_model.model.df["HETATM"][["x_coord", "y_coord", "z_coord"]]

                if len(true_model_df) != len(result_model_df):
                    continue

                rmsd = get_rmsd_dfs(true_model_df,
                                    result_model_df,
                                    )

                rmsds.append(rmsd)

            if len(rmsds) == 0:
                print("\tCOULD NOT COMPARE TO TRUE! SKIPPING!")
                continue

            candidate_model_record["rmsd"] = min(rmsds)

            candidate_model_results.append(candidate_model_record)

        if len(candidate_model_results) == 0:
            print("\tCOULD NOT COMPARE TO TRUE! SKIPPING!")
            continue

        record["min_rmsd"] = min([candidate_model_record["rmsd"]
                                  for candidate_model_record
                                  in candidate_model_results]
                                 )

        record["mean_rmsd"] = np.mean([candidate_model_record["rmsd"]
                                       for candidate_model_record
                                       in candidate_model_results]
                                      )

        record["time"] = result.time

        method_results.append(record)

    df = pd.DataFrame(method_results)

    return df


if __name__ == "__main__":
    print("Parsing args...")
    args = parse_args()

    print("Setting up configuration...")
    config: Config = get_config(args)

    print("Reading events csv...")
    df: pd.DataFrame = get_table(config.events_csv_path)
    print("\tGot events csv with: {} events".format(len(df)))

    print("Getting events...")
    events: List[Event] = get_events(df)
    print("\tGot: {} events".format(len(events)))

    print("Making ouput dirs...")
    events = make_output_dir(config.output_dir,
                             events,
                             )
    print("\tAfter filterning: {} events".format(len(events)))

    for event in events:
        print("Processing event: {} {}".format(event.dtag, event.event_idx))

        if not is_event_built(event):
            print("\tNo Model for event at: {}! Skipping!".format(event.final_model_path))
            continue

        # if event.dtag != "JMJD1BA-x1258":
        #     continue

        protein_model_path: Path = get_protein_model_path(event)
        ligand_model_path: Path = get_ligand_model_path(event)
        event_map_path: Path = get_event_map_path(event)
        resolution: float = event.analysed_resolution
        event_output_dir_path: Path = config.output_dir / "{}_{}".format(event.dtag,
                                                                         event.event_idx,
                                                                         )
        data_path: Path = event.data_path

        print("\tPlacing ligand...")
        placed_ligand_path = coarse_build(ligand_model_path=ligand_model_path,
                                          event=event,
                                          output_path=event_output_dir_path / "ligand.pdb",
                                          )

        print("\tStripping receptor waters...")
        stripped_receptor_path = strip_receptor_waters(receptor_path=protein_model_path,
                                                       placed_ligand_path=placed_ligand_path,
                                                       output_path=event_output_dir_path / "stripped_receptor.pdb",
                                                       )

        print("\tConverting event map to mtz...")
        event_map_mtz_path: Path = event_map_to_mtz(event_map_path,
                                                    event_output_dir_path / "{}_{}.mtz".format(event.dtag,
                                                                                               event.event_idx,
                                                                                               ),
                                                    event.analysed_resolution,
                                                    )

        # QFit location
        # result_qfit_table: pd.DataFrame = autobuild_event_qfit(protein_model_path,
        #                                                        ligand_model_path,
        #                                                        event_map_path,
        #                                                        resolution,
        #                                                        output_dir_path,
        #                                                        )
        # QFit Event map
        print("\tQfit on event map...")
        result_qfit: Result = autobuild_event_qfit(stripped_receptor_path,
                                                   placed_ligand_path,
                                                   event_map_path,
                                                   resolution,
                                                   event_output_dir_path,
                                                   )

        # Phenix
        # Phenix control
        result_phenix_control: Result = autobuild_event_phenix(protein_model_path=stripped_receptor_path,
                                                               ligand_model_path=ligand_model_path,
                                                               event_mtz_path=data_path,
                                                               output_dir_path=event_output_dir_path / "phenix_control",
                                                               event_centroid=None,
                                                               )

        # Phenix location
        # Phenix Event map
        result_phenix: Result = autobuild_event_phenix(protein_model_path=stripped_receptor_path,
                                                       ligand_model_path=placed_ligand_path,
                                                       event_mtz_path=event_map_mtz_path,
                                                       output_dir_path=event_output_dir_path / "phenix_event",
                                                       event_centroid=[event.x, event.y, event.z],
                                                       )
        # result_phenix: Result = autobuild_event_phenixevent)

        # Rhofit
        # Rhofit location
        # Rhofit Event map
        # result_qfit_table: pd.DataFrame = autobuild_event_buster(event)

        # Diff
        # Diff location
        # Diff Event map
        # result_qfit_table: pd.DataFrame = autobuild_event_diff(event)

        print("\tComparing to true model at: {}...".format(event.final_model_path))
        results_df: pd.DataFrame = get_results(results={"qfit_event_map": result_qfit,
                                                        "phenix_event_map": result_phenix,
                                                        "phenix_control": result_phenix_control,
                                                        },
                                               true_model_path=event.final_model_path,
                                               )

        print(results_df)
