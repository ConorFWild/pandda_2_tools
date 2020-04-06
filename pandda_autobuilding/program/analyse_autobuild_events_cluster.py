from typing import NamedTuple, Dict, List
import os
import logging
import subprocess
from functools import partial
import json

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
from autobuild.luigi import ProcessorLuigi
from autobuild.parallel import process_dask


def parse_args():
    parser = argparse.ArgumentParser()
    # IO

    parser.add_argument("-i", "--input_events_csv",
                        type=str,
                        help="The directory for output and intermediate files to be saved to",
                        required=True
                        )

    parser.add_argument("-a", "--autobuilding_dir_path",
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
    autobuilding_dir_path: Path
    output_dir: Path
    events_csv_path: Path


def get_config(args):
    config: Config = Config(autobuilding_dir_path=Path(args.autobuilding_dir_path),
                            output_dir=Path(args.output_dir_path),
                            events_csv_path=Path(args.input_events_csv),
                            )

    return config


def make_results_dataframe(all_results,
                           true_model_paths,
                           events_dict,
                           ):
    records = []

    for event_id, true_model_path in true_model_paths.items():
        print("\tAnalysing event for: {}".format(event_id))

        true_model = BioPandasModel(true_model_path)
        event = events_dict[event_id]

        results = all_results[event_id]

        method_results = []

        for build_name, result in results.items():
            record = {}
            record["dtag"] = event_id[0]
            record["event_idx"] = event_id[1]
            record["method"] = build_name
            record["num_candidates"] = len(result.result_model_paths)

            candidate_model_results = []
            distances_to_events = []

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

                    true_model_df = \
                        true_model.model.df["HETATM"][true_model.model.df["HETATM"]["chain_id"] == chain_id][
                            ["x_coord", "y_coord", "z_coord"]]
                    result_model_df = \
                        result_model.model.df["HETATM"][["x_coord", "y_coord", "z_coord"]]

                    true_model_mean_coords = np.mean(np.array(true_model_df), axis=0)
                    event_mean_coords = np.array([event.x,
                                                  event.y,
                                                  event.z,
                                                  ])
                    distance_from_event_to_model = np.linalg.norm(true_model_mean_coords - event_mean_coords)




                    if len(true_model_df) != len(result_model_df):
                        continue

                    rmsd = get_rmsd_dfs(true_model_df,
                                        result_model_df,
                                        )

                    rmsds.append(rmsd)

                    distances_to_events.append(distance_from_event_to_model)


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

        if len(method_results) == 0:
            print("\tCould not compare to true: skipping!")
            continue
        if len(distances_to_events) == 0:
            continue
        records.append(method_results)
        print(pd.DataFrame(method_results))
        print("True hit mean coordinates: {}".format(min(distances_to_events)))

    print(records)
    for record in records:
        if type(record) != type([]):
            print(record)
    df_records = sum(records)

    df = pd.DataFrame(df_records)

    return df


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


def get_table(event_csv_path: Path):
    return pd.read_csv(str(event_csv_path))


def is_event_built(event: Event):
    if event.actually_built:
        return True
    else:
        return False


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

    print("\tAfter filterning: {} events".format(len(events)))

    results = {}
    final_model_paths = {}

    for event in events:
        print("Processing event: {} {}".format(event.dtag, event.event_idx))

        if not is_event_built(event):
            print("\tNo Model for event at: {}! Skipping!".format(event.final_model_path))
            continue

        event_output_dir_path: Path = config.autobuilding_dir_path / "{}_{}".format(event.dtag,
                                                                                    event.event_idx,
                                                                                    )

        # Get Phenix control
        phenix_control_json_path: Path = event_output_dir_path / "phenix_control.json"
        try:
            with open(str(phenix_control_json_path), "r") as f:
                string = f.read()
                results_dict = json.loads(string)
                phenix_control_result = Result(time=float(results_dict["time"]),
                                               success=bool(results_dict["success"]),
                                               result_model_paths=[Path(p)
                                                                   for p
                                                                   in results_dict[
                                                                       "result_model_paths"]
                                                                   ],
                                               )
        except Exception as e:
            print(e)
            print("\tCouldn't find an results json! Skipping!")
            continue

        # Get Phenix event
        phenix_event_json_path: Path = event_output_dir_path / "phenix_event.json"
        try:
            with open(str(phenix_event_json_path), "r") as f:
                string = f.read()
                results_dict = json.loads(string)
                phenix_event_result = Result(time=float(results_dict["time"]),
                                             success=bool(results_dict["success"]),
                                             result_model_paths=[Path(p)
                                                                 for p
                                                                 in results_dict[
                                                                     "result_model_paths"]
                                                                 ],
                                             )

        except Exception as e:
            print(e)
            print("\tCouldn't find an results json! Skipping!")
            continue

        results[(event.dtag, event.event_idx)] = {}
        results[(event.dtag, event.event_idx)]["phenix_control"] = phenix_control_result
        results[(event.dtag, event.event_idx)]["phenix_event"] = phenix_event_result
        final_model_paths[(event.dtag, event.event_idx)] = event.final_model_path

    print("\tFinished getting results jsons, with: {} jsons found".format(len(results)))

    print("Making results dataframe")
    events_dict = {(event.dtag, event.event_idx): event
                   for event in events}
    results_df = make_results_dataframe(results,
                                        final_model_paths,
                                        events_dict,
                                        )
    print("\tMade results dataframe")

    print("Outputing results dataframe to: {}".format(config.output_dir / "results.csv"))
    results_df.to_csv(config.output_dir / "results.csv")
    print("\tOutput results dataframe")
