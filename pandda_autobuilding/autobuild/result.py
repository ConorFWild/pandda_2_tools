from typing import NamedTuple, List
from pathlib import Path

class Result(NamedTuple):
    result_model_paths: List[Path]
    success: bool
    time: float