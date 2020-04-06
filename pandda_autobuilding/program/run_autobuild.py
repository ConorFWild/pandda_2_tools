from typing import NamedTuple, Dict
import logging

import argparse

from autobuild.cmd import (autobuild_from_cmd,
                           )


def parse_args():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("-b", "--builder",
                        type=str,
                        help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                        required=True
                        )

    parser.add_argument("-p", "--pandda_dir_path",
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


if __name__ == "__main__":
    args = parse_args()

    autobuild_from_cmd(args)
