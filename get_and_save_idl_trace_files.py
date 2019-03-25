#!/usr/bin/env python

import numpy as np
import os
import glob
import argparse
import time

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

    n_polynomial_coefficients = int(trace[1].header['NPOLY'])

    x = np.arange(0, int(trace[1].header['NX']), pixel_sampling)

    # Apparently the Lengendre polynomials need to be evaluated -1 to 1
    normalized_x = (0.5 + x) / int(trace[1].header['NX']) - 0.5
    normalized_x *= 2.0

    # Make ds9 region file with the traces
    # Don't forget the ds9 is one indexed for pixel positions
    ds9_lines = ""
    trace_centers = {'id': [], 'centers': []}
    for fiber in fibers:
        for order in range(int(trace[1].header['NORD'])):
            coefficients = trace[1].data[0, fiber, order, :n_polynomial_coefficients]
            polynomial = np.polynomial.legendre.Legendre(coefficients)
            trace_center_positions = polynomial(normalized_x)
            trace_centers['id'].append(order)
            trace_centers['centers'].append(trace_center_positions)
    data = Table(trace_centers)
    hdu = fits.BinTableHDU(data, name='TRACE', header=trace[1].header)
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
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--site',
    #                    choices=['tlv', 'elp', 'lsc', 'cpt'],
    #                    help='NRES site name, e.g. tlv')
    #parser.add_argument('--archive-base-path',
    #                    help='archive data path, e.g. /archive/engineering/')
    #parser.add_argument('--unbiased-centers-path')
    #args = parser.parse_args()
    #site = args.site
    #unbiased_centers_path = args.unbiased_centers_path
    #archive_base_path = args.archive_base_path
    site = 'tlv'
    centers_path = '/tmp/trace_flux_centering_20190325'
    archive_base_path = '/home/mbrandt21/Documents/nres_archive_data'

    instrument = {'lsc': 'nres01', 'elp': 'nres02', 'cpt': 'nres03', 'tlv': 'nres04'}[site]
    unbiased_centers_path = os.path.join(centers_path, '{0}/'.format(site))
    output_folder = os.path.join(unbiased_centers_path, 'idl_centers')
    raw_data_basepath = os.path.join(archive_base_path, '{0}/{1}'.format(site, instrument))

    print(unbiased_centers_path)
    print(output_folder)
    print(raw_data_basepath)

    idl_trace_files = glob.glob(os.path.join(raw_data_basepath, '*/specproc/*trace_*'), recursive=True)
    unbiased_center_files = glob.glob(os.path.join(unbiased_centers_path, '*unbiased-trace*'))

    idl_trace_filedates = [extract_date_idl_filename(trace_file) for trace_file in idl_trace_files]
    unbiased_center_filedates = [extract_date_banzai_filename(cntrs_file) for cntrs_file in unbiased_center_files]

    print(idl_trace_files)
    print(unbiased_center_files)
    print(idl_trace_filedates)
    print(unbiased_center_filedates)

    time.sleep(1)
    matched_idl_trace_files = []
    for trace_file, file_date in zip(idl_trace_files, idl_trace_filedates):
        if file_date not in unbiased_center_filedates:
            continue
        matched_idl_trace_files.append(trace_file)
        hdu_list = extract_traces_from_idl_file(trace_file, fibers=[0, 1, 2], pixel_sampling=1)
        output_name = os.path.basename(trace_file).replace('trace', 'banzai_style_trace')
        hdu_list.writeto(os.path.join(output_folder, output_name),
                         output_verify='exception', overwrite=True)

    print(len(matched_idl_trace_files), len(unbiased_center_files))
