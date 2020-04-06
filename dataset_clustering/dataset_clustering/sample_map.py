import numpy as np

import mdc3
from mdc3.types.real_space import interpolate_uniform_grid


class Sampler:

    def __init__(self, xmap,
                 alignment_moving_to_ref,
                 res_centre_coords,
                 grid_params=[20, 20, 20],
                 offset=[10, 10, 10],
                 ):
        self.xmap = xmap
        self.alignment_moving_to_ref = alignment_moving_to_ref
        self.res_centre_coords = res_centre_coords
        self.grid_params = grid_params
        self.offset = offset

    def __call__(self):
        return sample_map(self.xmap,
                          self.alignment_moving_to_ref,
                          self.res_centre_coords,
                          self.grid_params,
                          self.offset,
                          )


def sample_map(xmap,
               alignment_moving_to_ref,
               res_centre_coords,
               grid_params=[20, 20, 20],
               offset=[10, 10, 10],
               ):
    # Align and Get RTop to moving protein frame from alignment

    # print("\tres_centre_coords: {}".format(res_centre_coords))

    moving_to_ref_translation = alignment_moving_to_ref.rotran[1]
    # print("\tmoving_to_ref_translation: {}".format(moving_to_ref_translation))

    rotation = alignment_moving_to_ref.rotran[0]
    # print("\talignment_moving_to_ref: {}".format(alignment_moving_to_ref))

    rotated_offset = np.matmul(rotation, offset)
    # print("\trotated_offset: {}".format(rotated_offset))

    translation = moving_to_ref_translation - res_centre_coords - rotated_offset
    # print("\tSampling around point: {}".format(translation))

    # Interpolate NX map in moving protein frame
    nxmap = interpolate_uniform_grid(xmap,
                                     translation,
                                     np.transpose(rotation),
                                     grid_params=grid_params,
                                     )
    nxmap_data = nxmap.export_numpy()

    return nxmap_data
