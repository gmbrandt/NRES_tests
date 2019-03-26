#!/usr/bin/env python

import numpy as np
import os
import glob
import argparse
import time
import tempfile

from astropy.io import fits
from astropy.table import Table


def open_fits_file(filename):
    """
    Load a fits file

    Parameters
    ----------
    filename: str
              File name/path to open

    Returns
    -------
    hdulist: astropy.io.fits

    Notes
    -----
    This is a wrapper to astropy.io.fits.open but funpacks the file first.
    """
    base_filename, file_extension = os.path.splitext(os.path.basename(filename))
    if file_extension == '.fz':
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_filename = os.path.join(tmpdirname, base_filename)
            os.system('funpack -O {0} {1}'.format(output_filename, filename))
            hdulist = fits.open(output_filename, 'readonly')
    else:
        hdulist = fits.open(filename, 'readonly')

    return hdulist


def sort_traces(traces_dict):
    center = int(traces_dict['centers'].shape[1] / 2)
    traces_dict['centers'] = traces_dict['centers'][traces_dict['centers'][:, center].argsort()]
    traces_dict['id'] = np.arange(traces_dict['centers'].shape[0])
    return traces_dict


def extract_traces_from_idl_file(trace_file, fibers, pixel_sampling=1, sort=True):
    """
    This is adopted from https://github.com/LCOGT/nres-pipe/blob/master/nrespipe/traces.py
    :return: HDU with trace centers in the same fashion that banzai_nres does.
    """
    # TODO: Good metrics could be total flux in extraction region for the flat after subtracting the bias.
    # TODO: Average S/N per x-pixel (summing over the profile doing an optimal extraction)

    # read in the trace file
    trace = open_fits_file(trace_file)

    n_polynomial_coefficients = int(trace[0].header['NPOLY'])

    x = np.arange(0, int(trace[0].header['NX']), pixel_sampling)

    # Legendre polynomials need to be evaluated -1 to 1
    normalized_x = (0.5 + x) / int(trace[0].header['NX']) - 0.5
    normalized_x *= 2.0

    trace_centers = {'id': [], 'centers': []}
    for fiber in fibers:
        for order in range(int(trace[0].header['NORD'])):
            coefficients = trace[0].data[0, fiber, order, :n_polynomial_coefficients]
            polynomial = np.polynomial.legendre.Legendre(coefficients)
            trace_center_positions = polynomial(normalized_x)
            trace_centers['id'].append(order)
            trace_centers['centers'].append(trace_center_positions)
    for key, item in trace_centers.items():
        trace_centers[key] = np.array(item)
    if sort:
        trace_centers = sort_traces(traces_dict=trace_centers)
    data = Table(trace_centers)
    hdu = fits.BinTableHDU(data, name='TRACE', header=trace[0].header)
    hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
    return hdu_list


def extract_date_idl_filename(filepath):
    filename = os.path.basename(filepath)
    filename_no_ext = filename.split('.fits')[0]
    date_string = filename_no_ext.split('_')[-1]
    return date_string


def extract_date_banzai_filename(filepath):
    filename = os.path.basename(filepath)
    filename_no_ext = filename.split('.fits')[0]
    date_string = filename_no_ext.split('-')[2]
    return date_string


if __name__ == "__main__":
    # erroneous directory where IDL pipeline is putting the CPT data:
    # /archive/engineering/cpt/nres03/20181205/specproc/trace_cpt_nres03_fa13_20190324.fits.fz
    # so we do a very large glob.glob search for these files.
    parser = argparse.ArgumentParser()
    parser.add_argument('--site',
                        choices=['tlv', 'elp', 'lsc', 'cpt'],
                        help='NRES site name, e.g. tlv')
    parser.add_argument('--unbiased-centers-path',
                        help='path to the unbiased center files, e.g. /archive/engineering/')
    parser.add_argument('--archive-base-path',
                        help='archive data path, e.g. /archive/engineering/')
    parser.add_argument('--process-idl-masters',
                        help='munges the idl files into 110 fiber formats', action="store_true")
    args = parser.parse_args()
    site = args.site
    unbiased_centers_path = args.unbiased_centers_path
    archive_base_path = args.archive_base_path
    #site = 'tlv'
    #unbiased_centers_path = '/tmp/trace_flux_centering_20190325'
    #archive_base_path = '/home/mbrandt21/Documents/nres_archive_data'

    instrument = {'lsc': 'nres01', 'elp': 'nres02', 'cpt': 'nres03', 'tlv': 'nres04'}[site]
    unbiased_centers_path = os.path.join(unbiased_centers_path, '{0}/'.format(site))
    output_folder = os.path.join(unbiased_centers_path, 'idl_centers', site, instrument, 'many_dates', 'processed')
    raw_data_basepath = os.path.join(archive_base_path, '{0}/{1}'.format(site, instrument))

    idl_trace_files = glob.glob(os.path.join(raw_data_basepath, '*/specproc/*trace_*'), recursive=True)
    unbiased_center_files = glob.glob(os.path.join(unbiased_centers_path, '*unbiased-trace*'))

    idl_trace_filedates = [extract_date_idl_filename(trace_file) for trace_file in idl_trace_files]
    unbiased_center_filedates = [extract_date_banzai_filename(cntrs_file) for cntrs_file in unbiased_center_files]

    matched_idl_trace_files = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for trace_file, file_date in zip(idl_trace_files, idl_trace_filedates):
        if file_date not in unbiased_center_filedates:
            continue
        matched_idl_trace_files.append(trace_file)
        hdu_list = extract_traces_from_idl_file(trace_file, fibers=[0, 1], pixel_sampling=1, sort=True)
        output_name = os.path.basename(trace_file).replace('trace', 'banzai_style_trace_110')
        hdu_list.writeto(os.path.join(output_folder, output_name),
                         output_verify='exception', overwrite=True)

