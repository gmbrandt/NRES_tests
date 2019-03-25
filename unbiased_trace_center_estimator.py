#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.table import Table
import glob
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import sep
import os
import argparse


"""
Script which compares the outputs of tracing to the actual centroids of the traces: The mean of
the position, weighted by flux.
"""


def unbiased_trace_centers(image_hdu, trace_hdu, halfwindow=5):
    trace_centers = trace_hdu['TRACE'].data['centers']
    image_data = {'counts': image_hdu[1].data.astype(np.float32)}
    image_data['counts'] -= sep.Background(image_data['counts']).back()
    image_data['x_coords'], image_data['y_coords'] = np.meshgrid(np.meshgrid(np.arange(image_data['counts'].shape[1])),
                                                                 np.arange(image_data['counts'].shape[0]))
    trace_center_estimates = []
    trace_center_estimate_errors = []
    for single_order_centers in trace_centers:
        flux_weighted_centers, errors = estimate_trace_centers(single_order_centers, image_data, halfwindow=halfwindow)
        trace_center_estimates.append(flux_weighted_centers)
        trace_center_estimate_errors.append(errors)
    return np.array(trace_center_estimates), np.array(trace_center_estimate_errors)


def estimate_trace_centers(trace_center_positions, image_data, halfwindow=5):
    mask = close_to_trace_mask(trace_center_positions, image_data, halfwindow=halfwindow).astype(np.float32)
    min_y, max_y = np.min(trace_center_positions) - halfwindow - 2, np.max(trace_center_positions) + halfwindow + 2
    min_row, max_row = max(int(min_y), 0), min(int(max_y), image_data['counts'].shape[0])
    weights = mask[min_row:max_row] * image_data['counts'][min_row:max_row]
    trace_positions = np.average(image_data['y_coords'][min_row:max_row], weights=weights, axis=0)
    xminusxavg = image_data['y_coords'][min_row:max_row] - trace_positions
    standard_deviations = np.sqrt(np.sum(weights * xminusxavg ** 2, axis=0) /
                                  np.sum(weights**2, axis=0))
    trace_positions[weights.sum(axis=0) < 1000] = np.nan
    standard_deviations[weights.sum(axis=0) < 1000] = np.nan
    return trace_positions, standard_deviations


def close_to_trace_mask(trace_center_positions, image_data, halfwindow=5):
    max_y, min_y = image_data['counts'].shape[0] - 1, 0
    mask = np.zeros_like(image_data['counts'], dtype=np.int)
    x_pixels = np.arange(mask.shape[1])
    y_pixels = np.ones((halfwindow*2+1, image_data['counts'].shape[1]))
    y_pixels *= trace_center_positions
    y_pixels += np.array([np.arange(-halfwindow, halfwindow+1)]).T
    y_pixels[y_pixels > max_y] = max_y
    y_pixels[y_pixels < min_y] = min_y
    for near_trace in y_pixels.astype(np.int):
        mask[near_trace, x_pixels] = 1
    return mask


def mask_for_pixels_close_to_trace(trace_center_positions, image_y_coordinate_array, halfwindow=5):
    mask = np.zeros_like(image_y_coordinate_array)
    for j in range(image_y_coordinate_array.shape[1]):
        mask[:, j] = np.isclose(image_y_coordinate_array[:, j],
                                trace_center_positions[j], atol=halfwindow)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--site',
                        choices=['tlv', 'elp', 'lsc', 'cpt'],
                        help='NRES site name, e.g. tlv')
    parser.add_argument('--output-base-path')
    parser.add_argument("--plot", help="plot trace comparison data",
                        action="store_true")
    parser.add_argument("--calculate", help="calculate flux weighted mean of trace centers",
                        action="store_true")
    parser.add_argument('--archive-base-path',
                        help='archive data path, e.g. /archive/engineering/')
    args = parser.parse_args()
    site = args.site
    output_dir = os.path.join(args.output_base_path, '{0}/'.format(site))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    instrument = {'lsc': 'nres01', 'elp': 'nres02', 'cpt': 'nres03', 'tlv': 'nres04'}[site]
    raw_data_basepath = os.path.join(args.archive_base_path, '{0}/{1}'.format(site, instrument))

    all_master_traces = glob.glob(os.path.join(raw_data_basepath, '**/processed/*trace*'))

    if args.calculate:
        for master_trace in all_master_traces:

            master_dark, master_bias = '/tmp/none1111.fits', '/tmp/none1111.fits'
            if '-110' in master_trace:
                master_dark = (master_trace.replace('trace', 'dark')).replace('-110', '')
                master_bias = (master_trace.replace('trace', 'bias')).replace('-110', '')
            if '-011' in master_trace:
                master_dark = (master_trace.replace('trace', 'dark')).replace('-011', '')
                master_bias = (master_trace.replace('trace', 'bias')).replace('-011', '')
            if not os.path.exists(master_dark) or not os.path.exists(master_bias):
                # short circuit if master bias or master darks were not taken.
                continue

            master_flat = master_trace.replace('trace', 'lampflat')
            output_filename = os.path.basename(master_trace).replace('trace', 'unbiased-trace')
            output_path = os.path.join(output_dir, output_filename)

            traces_hdu = fits.open(master_trace)
            flat_hdu = fits.open(master_flat)
            flux_centers, errors = unbiased_trace_centers(flat_hdu, traces_hdu, halfwindow=7)

            data = Table({'centers': flux_centers, 'errors': errors})
            hdu = fits.BinTableHDU(data, name='TRACE')
            hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdu_list.writeto(output_path, output_verify='exception', overwrite=True)

    if args.plot:
        avg_locations = {}
        avg_locations_err = {}
        times = []
        low, high = 1500, 3000
        orders = np.arange(23, 26)
        for master_trace in all_master_traces:
            if '110' in master_trace:  # plotting only 110 trace files.
                traces_hdu = fits.open(master_trace)
                trace_centers = traces_hdu['TRACE'].data['centers']
                unbiased_path = os.path.join(output_dir,
                                             os.path.basename(master_trace).replace('trace', 'unbiased-trace'))
                if not os.path.exists(unbiased_path):
                    continue
                unbiased_centers_hdu = fits.open(unbiased_path)
                unbiased_centers = unbiased_centers_hdu['TRACE'].data['centers']
                residuals = unbiased_centers - trace_centers
                errors = unbiased_centers_hdu['TRACE'].data['errors']

                for order in orders:
                    if avg_locations.get(str(order)) is None:
                        avg_locations[str(order)] = []
                        avg_locations_err[str(order)] = []
                    avg_locations[str(order)].append(np.nanmean(unbiased_centers[order][low:high]))
                    included_points = np.count_nonzero(~np.isnan(errors[order][low:high]))
                    avg_locations_err[str(order)].append(np.sqrt(np.nansum(errors[order][low:high]**2)/included_points))

                times.append(datetime.datetime.strptime(traces_hdu[1].header['DATE-OBS'].split('.')[0],
                                                        '%Y-%m-%dT%H:%M:%S'))
        plt.figure()
        for order in orders:
            plt.errorbar(times, avg_locations[str(order)] - np.mean(avg_locations[str(order)]),
                         yerr=avg_locations_err[str(order)], marker=',', ls='none')
        plt.ylabel('Order position - average')
        plt.xticks(rotation='vertical')
        plt.show()
"""
x = np.arange(residuals.shape[1])
i = 0
min_order = 0
max_order = np.nan
max_order = min(residuals.shape[0] - 1, max_order)
for single_order_residuals, so_errors in zip(residuals[min_order:max_order], errors[min_order:max_order]):
    fig, ax = plt.subplots()
    ax.errorbar(x, single_order_residuals, yerr=so_errors)
    ax.set_xlabel('Pixel')
    ax.axhline(y=0, color='k')
    ax.set_title('BanzaiNRES Trace Center - Flux weighted Mean Center \n Trace ID: {0}'.format(i + min_order))
    i += 1
    plt.show()
"""
