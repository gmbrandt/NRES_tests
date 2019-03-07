#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sep
import math

"""
Script which compares the outputs of tracing to the actual centroids of the traces: The mean of
the position, weighted by flux.
"""


def banzai_trace_center_residuals(image_hdu, trace_hdu, halfwindow=5):
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

    #trace_center_estimates = np.array(trace_center_estimates)
    #trace_center_estimate_errors = np.array(trace_center_estimate_errors)
    #plt.figure()
    #plt.imshow(image_data['counts'])
    ##plt.plot(trace_centers[67], 'b')
    #for order in range(trace_centers.shape[0]):
    #    plt.plot(trace_center_estimates[order], 'r')
    #plt.show()
    residuals = trace_centers - trace_center_estimates
    return residuals, trace_center_estimate_errors


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
    traces_hdu = fits.open('/home/mbrandt21/Documents/nres_archive_data/'
                           'tlv/nres04/20190208/processed/tlvnrs04'
                           '-fa18-20190208-trace-bin1x1-110.fits.fz')
    raw_image_hdu = fits.open('/home/mbrandt21/Documents/nres_archive_data/'
                              'tlv/nres04/20190208/processed/tlvnrs04'
                              '-fa18-20190208-lampflat-bin1x1-110.fits.fz')
    residuals, errors = banzai_trace_center_residuals(raw_image_hdu, traces_hdu, halfwindow=7)
    x = np.arange(residuals.shape[1])
    i = 0
    min_order = 100
    max_order = 130
    for single_order_residuals, so_errors in zip(residuals[min_order:max_order], errors[min_order:max_order]):
        fig, ax = plt.subplots()
        ax.errorbar(x, single_order_residuals, yerr=so_errors)
        ax.set_xlabel('Pixel')
        ax.axhline(y=0, color='k')
        ax.set_title('BanzaiNRES Trace Center - Flux weighted Mean Center \n Trace ID: {0}'.format(i + min_order))
        i += 1
        plt.show()
