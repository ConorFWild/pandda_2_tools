import os
from pathlib import Path

import pandas as pd


def parse_for_regex(path, regex):
    paths = path.glob(regex)


def parse_pandda_pdbs(pandda_dir_path,
                      events_dict,
                      ):
    pdb_regex = "{dtag}-pandda-input.pdb"

    pandda_pdbs = {event_id: pandda_dir_path / "processed_datasets" / event.dtag / pdb_regex.format(dtag=event.dtag)
                   for event_id, event
                   in events_dict.items()
                   }

    return pandda_pdbs


def parse_pandda_event_maps(pandda_dir_path, events_dict):
    event_map_regex = "{dtag}-event_{site}_{event_idx}-BDC_{bdc}_map.native.ccp4"

    pandda_ccp4s = {
        event_id: pandda_dir_path / "processed_datasets" / event.dtag / event_map_regex.format(dtag=event.dtag,
                                                                                               site=event.site,
                                                                                               event_idx=event.event_idx,
                                                                                               bdc=event.bdc,
                                                                                               )
        for event_id, event
        in events_dict.items()
        }

    return pandda_ccp4s


def parse_ligand_pdbs(pandda_dir_path, events_dict):
    compound_regex = "ligand_files/*.pdb"

    compound_pdbs = {}

    for event_id, event in events_dict.items():
        processed_dataset_path = pandda_dir_path / "processed_datasets" / event.dtag
        ligands = list((processed_dataset_path / "ligand_files").glob("*.pdb"))
        ligand_strings = [str(ligand_path) for ligand_path in ligands if ligand_path.name != "tmp.pdb"]

        ligand_pdb_path = Path(min(ligand_strings,
                                   key=len,
                                   )
                               )
        compound_pdbs[event_id] = ligand_pdb_path

    return compound_pdbs


def parse_pandda_event_table(pandda_dir_path):
    pandda_event_table_path = pandda_dir_path / "analyses/pandda_analyse_events.csv"
    event_table = pd.read_csv(str(pandda_event_table_path))

    return event_table


def get_resolutions(events_dict):
    return {event_id: event.resolution for event_id, event in events_dict.items()}


def make_output_dirs(output_dir, event_dict):
    output_dirs = {}

    try:
        os.mkdir(str(output_dir))
    except:
        pass

    for event_idx, event in event_dict.items():
        try:
            os.mkdir(str(output_dir / "{}_{}".format(event.dtag, event.event_idx)))
        except:
            pass

        output_dirs[event_idx] = output_dir / "{}_{}".format(event.dtag, event.event_idx)

    return output_dirs

def output_results_table(results_table, output_dir):
    results_table.to_csv(output_dir / "results.csv")
