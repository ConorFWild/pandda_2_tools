import numpy as np

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration,
                                 )
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D,
                                   )

def align_xmap_np(static, moving, static_grid2world=np.eye(4), moving_grid2world=np.eye(4)):
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors,
                                verbosity=0)
    transform = RigidTransform3D()
    params0 = None

    # starting_affine = transform_centers_of_mass(static,
    #                                             static_grid2world,
    #                                             moving,
    #                                             moving_grid2world,
    #                                             ).affine
    # print("\tCOM affine transform is: {}".format(starting_affine))

    starting_affine = np.eye(4)



    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine,

                            # verbosity=0,
                            )

    # Transform
    transformed = rigid.transform(moving)
    # print("\tThe transform is: {}".format(rigid.affine))

    return transformed
