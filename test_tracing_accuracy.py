#!/usr/bin/env python

import numpy as np
from astropy.io import fits

"""
Script which compares the outputs of tracing to the actual centroids of the traces: The mean of
the position, weighted by flux.
"""
