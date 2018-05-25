import numpy as np
import os
import re
import errno
import logging

def betti_sum_bounds(dims, degree):
    """
    The upper bound on the sum of all betti numbers for a polynomial of certain
    dimensions and degree
    """
    return degree*(2*degree -1.0)**(dims - 1.0)

def seed_rng():
    # Random seed stuff
    import datetime
    import random
    np.random.seed(datetime.datetime.now().microsecond)
    random.seed(datetime.datetime.now().microsecond)

def create_dir_if_not_exist(dirname):

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno == errno.EEXIST and os.path.isdir(dirname):
                pass
            else:
                raise
        logging.info('Created directory %s', dirname)
    else:
        logging.info('Directory %s already exists', dirname)

    return 1