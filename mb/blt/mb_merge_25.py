#!/usr/bin/env python
# coding: utf-8

# Merge multibeam bathymetry with Smith & Sandwell.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata, NearestNDInterpolator
from scipy.signal import medfilt2d
from scipy import ndimage
from munch import Munch, munchify
import xarray as xr
import gvpy as gv


# load data
def load_data(file):
    """
    Load BLT bathymetry data.

    Parameters
    ----------
    file : str
        Path and file name for processed multibeam data

    Returns
    -------
    d : xarray Dataset
        Multibeam data
    """
    # load multibeam data
    print('loading multibeam')
    d = xr.open_dataarray(file)

    # transform from DataArray to Dataset
    d = xr.Dataset({'mb': d})

    # mask for all mb data
    allmb = np.ones_like(d.mb)
    allmb[np.isnan(d.mb)] = np.nan
    d['allmb'] = (['y', 'x'], allmb)

    d['qc'] = (['y', 'x'], d['mb'].data.copy())

    return d


def clean_data(d, config, res):
    # reject mb data where difference between ss and mb is too large
    if 0:
        topodiff = np.absolute(d.mb - d.ss)
        badabs = topodiff > config.max_diff
        d.qc.data[badabs] = np.nan
        d['badabs'] = (['y', 'x'], badabs)

    # reject mb data where gradient is too large
    gradx, grady = np.gradient(d.qc, res)
    with np.errstate(invalid='ignore'):   # suppres warning because of NaNs
        badgrad = np.absolute(gradx) + np.absolute(grady) > config.max_slope
    d.qc.data[badgrad] = np.nan
    d['badgrad'] = (['y', 'x'], badgrad)

    # reject data where gradient changes sign like crazy
    gradD = np.sqrt(gradx**2 + grady**2)
    doublegradx, doublegrady = np.gradient(gradD)
    doublegradD = np.sqrt(doublegradx**2 + doublegrady**2)
    with np.errstate(invalid='ignore'):   # suppres warning because of NaNs
        baddoublegrad = doublegradD > config.max_double_slope
    d.qc.data[baddoublegrad] = np.nan
    d['baddoublegrad'] = (['y', 'x'], baddoublegrad)

    # reject data without neighbors
    # we'll use one of scipy's morphology functions in ndimage
    # http://www.scipy-lectures.org/advanced/image_processing/
    a = ~np.isnan(d.qc.data)
    aa = ndimage.binary_opening(a, structure=np.ones((3, 3)))
    aa = aa.astype('bool')
    d.qc.data[~aa] = np.nan
    badsinglepoints = a.astype('int') + aa.astype('int') == 1
    d['badsinglepoints'] = (['y', 'x'], badsinglepoints)

    return d


def plot_mask(mask, ax=None, **kwargs):
    # plotting function for rejected values
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.imshow(mask, origin='lower', vmin=0, vmax=1, cmap='gray_r')
    ax.set(**kwargs)


def plot_rejected(d):
    # plot rejected values
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 7.5),
                           sharex=True, sharey=True)
    plot_mask(d.allmb, ax[0, 0], title='allmb')
    if 0:
        plot_mask(d.badabs, ax[0, 1], title='bad absolute value')
    plot_mask(d.badgrad, ax[0, 2], title='bad gradient')
    plot_mask(d.baddoublegrad, ax[1, 2], title='bad double gradient')
    plot_mask(d.badsinglepoints, ax[1, 1], title='bad single points')
    ax[1, 0].set_axis_off()
#    fig.suptitle('multibeam data rejection')
    plt.tight_layout()
    import os
    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.savefig('fig/mb_rejected_data.png', dpi=200, bbox_inches='tight')

    # print number of rejected values
    npts = {'all': count_valid_data_points(d.allmb.data),
            # 'badabs_pts': count_valid_data_points(d.badabs),
            'badgrad_pts': count_valid_data_points(d.badgrad),
            'baddoublegrad_pts': count_valid_data_points(d.baddoublegrad),
            'badsingle_pts': count_valid_data_points(d.badsinglepoints)}
    all_removed = 0
    for k, n in npts.items():
        print('{}: {}'.format(k, n))
        if k != 'all':
            all_removed += n
    print('total points removed: {}'.format(all_removed))
    print('  or {:1.1f}%'.format(all_removed / npts['all'] * 100))


def merge_mb_ss(d, config):
    merged = d.qc.data.copy()
    badqc = np.isnan(merged)
    merged[badqc] = d.ss.data[badqc]
    topodiff = np.absolute(d.mb - d.ss)
    badabs2 = topodiff > config.reject_ss
    combined_mask = np.logical_and(badabs2, badqc)
    merged[combined_mask] = np.nan
    # now inpaint regions with NaN's (where mb and ss differed too much)
    mask = ~np.isnan(merged)
    xx, yy = np.meshgrid(d.x, d.y)
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    data0 = np.ravel(merged[mask])
    print('interpolating - will take a minute')
    # generate interpolator
    interp0 = NearestNDInterpolator(xym, data0)
    # interpolate
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    d['merged'] = (['y', 'x'], result0)
    return d


def lowpassconv(data, win1, win2, nanfix=True):
    # low pass convolution
    from scipy.signal import convolve
    A = np.ones([win1, win2]) / (win1 * win2)
    y = convolve(data, A, 'same')
    if nanfix:
        mask = np.isnan(y)
        y[mask] = data[mask]
    return y


def diffuse_mb_ss(d, b):
    # Diffuse multibeam data into Smith & Sandwell
    Dmb = d.qc.data.copy()
    Dout = d.merged.data.copy()

    niter = np.int(np.ceil(0.5 * np.diff(b.lon[0:2]) / np.diff(d.x[0:2])))
    good = ~np.isnan(Dmb)

    for ii in range(niter):
        print('diffusing multibeam, iteration {} of {}'.format(ii + 1, niter))
        Dout = lowpassconv(Dout, 3, 3)
        Dout[good] = Dmb[good]
    d['merged'] = (['y', 'x'], Dout)
    return d


def median_filter(d, config):
    # Median filter
    mf = medfilt2d(d.merged.data, kernel_size=np.int(config.smoo_after_diff))
    d['merged'] = (['y', 'x'], mf)
    return d


def count_valid_data_points(data):
    if data.dtype == 'bool':
        return np.int16(data).sum()
    else:
        return np.int16(np.isfinite(data)).sum()


def remove_points(da, lonr, latr, condition):
    lonm, latm = np.meshgrid(da.x.data, da.y.data)
    xim = ((lonm > np.min(lonr)) & (lonm < np.max(lonr)))
    yim = ((latm > np.min(latr)) & (latm < np.max(latr)))
    allc = condition & xim & yim
    aad = da.data.copy()
    aad[allc] = np.nan
    return aad


if __name__ == '__main__':

    print('running mb processing')

    # set processing parameters
    name = 'blt_canyon_25m.nc'
    cwd = os.getcwd()
    file = os.path.join(cwd, 'grd', name)

    res = 25

    config = {'max_diff': 1000,
              'max_slope': 2.0,
              'max_double_slope': 0.8,
              'reject_neighbours': 5,
              'reject_ss': 200,
              'smoo_after_diff': np.max([3, np.ceil(300 / res)])}
    config = munchify(config)

    d = load_data(file)

    d = clean_data(d, config, res)
    plot_rejected(d)
    # d = merge_mb_ss(d, config)
    # d = diffuse_mb_ss(d, b)
    # d = median_filter(d, config)

    # save output
    print('saving output')
    out = Munch()
    out.name = 'blt_canyon_mb_25_qc.nc'
    cwd = os.getcwd()
    out.file = os.path.join(cwd, 'final', out.name)
    bm = xr.DataArray(d.qc.data,
                      coords={'lat': (['lat'], d.y.data),
                              'lon': (['lon'], d.x.data)},
                      dims=['lat', 'lon'],
                      name='z')
    bm.to_netcdf(out.file)
