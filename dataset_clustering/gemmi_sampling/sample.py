import numpy as np
import gemmi


def interpolate_xmap(xmap,
                     location,
                     rotation,
                     shape,
                     scale,
                     ):
    points_to_sample = get_sample_points(location,
                                         rotation,
                                         shape,
                                         scale,
                                         )

    sample_grid = sample(xmap,
                         points_to_sample,
                         shape,
                         )

    return sample_grid


def sample(xmap_grid,
           points_to_sample,
           shape,
           ):
    sample_grid = np.zeros(shape)

    for index, position in points_to_sample.items():
        sample_grid[index] = xmap_grid.interpolate_value(gemmi.Position(position))

    return sample_grid


def get_sample_points(location,
                      rotation,
                      shape,
                      scale,
                      ):
    points_to_sample = {}

    for x, y, z in zip(range(shape[0]),
                       range(shape[1]),
                       range(shape[2]),
                       ):
        reference_vector = scale * np.array([x, y, z])

        rotated_vector = np.matmul(rotation, reference_vector)

        position = location + rotated_vector

        points_to_sample[(x, y, z,)] = gemmi.Position(position[0],
                                                      position[1],
                                                      position[2],
                                                      )

    return points_to_sample
