import numpy as np
from astropy.io import fits
from astropy.table import Table


def extract_traces_from_idl_file(trace_file, fibers, pixel_sampling=1):
    """
    This is adopted from https://github.com/LCOGT/nres-pipe/blob/master/nrespipe/traces.py
    :return: HDU with trace centers in the same fashion that banzai_nres does.
    """
    # TODO: Good metrics could be total flux in extraction region for the flat after subtracting the bias.
    # TODO: Average S/N per x-pixel (summing over the profile doing an optimal extraction)

    # read in the trace file
    trace = fits.open(trace_file)

    n_polynomial_coefficients = int(trace[0].header['NPOLY'])

    x = np.arange(0, int(trace[0].header['NX']), pixel_sampling)

    # Apparently the Lengendre polynomials need to be evaluated -1 to 1
    normalized_x = (0.5 + x) / int(trace[0].header['NX']) - 0.5
    normalized_x *= 2.0

    # Make ds9 region file with the traces
    # Don't forget the ds9 is one indexed for pixel positions
    ds9_lines = ""
    trace_centers = {'id': [], 'centers': []}
    for fiber in fibers:
        for order in range(int(trace[0].header['NORD'])):
            coefficients = trace[0].data[0, fiber, order, :n_polynomial_coefficients]
            polynomial = np.polynomial.legendre.Legendre(coefficients)
            trace_center_positions = polynomial(normalized_x)
            trace_centers['id'].append(order)
            trace_centers['centers'].append(trace_center_positions)
    data = Table(trace_centers)
    hdu = fits.BinTableHDU(data, name='TRACE', header=trace[0].header)
    hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
    return hdu_list
