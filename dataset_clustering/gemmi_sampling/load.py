import gemmi


def get_grid_from_file(mtz_path,
                       columns=["FTW", "PHWT"],
                       sample_rate=4,
                       ):
    mtz = gemmi.read_mtz_file(str(mtz_path))

    grid = mtz.transform_f_phi_to_map(columns[0],
                                      columns[1],
                                      sample_rate=sample_rate,
                                      )

    return grid


def get_grids_from_paths(paths):
    grids = {}

    for dtag, path in paths:
        grids[dtag] = get_grid_from_file(path)

    return grids
