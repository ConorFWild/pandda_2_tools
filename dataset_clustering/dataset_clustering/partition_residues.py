from typing import Dict, Tuple

import pprint

from mdc3.types.datasets import Dataset
from Bio.PDB.Residue import Residue

def partition_residues(structures: Dict[str, Dataset],
                       length=3,
                       ) -> Dict[Tuple[int], Dict[str, Residue]]:
    # pp = pprint.PrettyPrinter(indent=4)

    reference_key = list(structures.keys())[0]

    chain_generators = {dtag: [chain for chain in structure.get_chains()]
                        for dtag, structure
                        in structures.items()
                        }

    # print("Chain generators")
    # pp.pprint(chain_generators)

    reference_chain = chain_generators[reference_key]

    chains = {}
    for i, chain in enumerate(reference_chain):

        chains_i = {}
        for dtag, chain_generator in chain_generators.items():
            try:
                chains_i[dtag] = chain_generator[i]
            except:
                pass

        chains[i] = chains_i

    # print("Chains")
    # pp.pprint(chains)

    partitioning = {}
    for i, chain in chains.items():
        chain_partition = partition_chain(chain,
                                          prefix=i,
                                          length=length,
                                          )
        partitioning.update(chain_partition)

    return partitioning


def partition_chain(chains, prefix=0, length=3, selection="CA"):

    # pp = pprint.PrettyPrinter(indent=4)


    # Get thre reference chain
    reference_chain_dtag = list(chains.keys())[0]

    # Make the residue dicts
    res_dicts = {dtag: {res.get_id()[1]: res[selection] for res in chain.get_residues() if indicator(res,
                                                                                                     selection,
                                                                                                     )
                        }
                 for dtag, chain
                 in chains.items()
                 }
    # print("Res dicts")
    # pp.pprint(res_dicts)

    # Get the dictionary of contiguous keys of length x
    contiguous_keys = {dtag: get_contiguous_keys(res_dict,
                                                 length=length,
                                                 )
                       for dtag, res_dict
                       in res_dicts.items()
                       }
    # print("Contiguous keys")
    # pp.pprint(contiguous_keys)

    # Match the keys
    partitioning = {}
    for contiguous_key in contiguous_keys[reference_chain_dtag].keys():
        partitioning[contiguous_key] = {}
        for dtag, res_dict in res_dicts.items():
            try:
                contiguous_ress = {key: res_dicts[dtag][key] for key in contiguous_key}
                partitioning[contiguous_key][dtag] = contiguous_ress
            except:
                print("\tDataset {} does not have matches for key set {}".format(dtag,
                                                                                 contiguous_key,
                                                                                 )
                      )

    return partitioning


def get_contiguous_keys(res_dict, length=3):
    contiguous_keys = {}
    for res_id, res in res_dict.items():

        keys = [res_id + i for i in range(length)]
        keys_in_dict = [(key in res_dict) for key in keys]

        if all(keys_in_dict):
            contiguous_ress = [res_dict[key] for key in keys]
            contiguous_keys[tuple(keys)] = contiguous_ress

    return contiguous_keys


def indicator(res, name):
    try:
        ca = res[name]
        return True
    except:
        return False
