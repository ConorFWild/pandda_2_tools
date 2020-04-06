from typing import NamedTuple, Dict
import argparse
from pathlib import Path

from autobuild.qfit import autobuild_qfit


class AutobuildCMDParams(NamedTuple):
    builder: str
    pandda_dir_path: Path
    output_dir_path: Path


def autobuild_from_cmd(args):
    params = AutobuildCMDParams(builder=args.builder,
                                pandda_dir_path=Path(args.pandda_dir_path),
                                output_dir_path=Path(args.output_dir_path))

    print(params)

    if params.builder == "qfit":
        autobuild_qfit(params.pandda_dir_path,
                       params.output_dir_path,
                       )
