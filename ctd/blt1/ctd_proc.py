#!/usr/bin/env python
# coding: utf-8

# # BLT1 CTD Processing

import sys
from pathlib import Path
import warnings
import numpy as np

import gvpy as gv
import ctdproc as ctd

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='no interpolation*')

def proc_raw(stn):
    ctd_raw = Path('/Users/gunnar/Projects/blt/data/blt1/ctd/raw')
    file = f'DY132_CTD{stn:03d}.hex'
    hexfile = ctd_raw.joinpath(file)
    print('Converting hex file...')
    # c = ctd.io.CTDHex(hexfile)
    # cx = c.to_xarray()
    cx = ctd.io.CTDx(hexfile)
    print('Cleaning data...')
    cx_ud = ctd.proc.run_all(cx)
    print('Depth binning...')
    dz = 1
    zmin = 5
    zmax = np.floor(cx_ud['down'].depth.max().data)
    datad1m = ctd.proc.bincast(cx_ud['down'], dz, zmin, zmax)
    datau1m = ctd.proc.bincast(cx_ud['up'], dz, zmin, zmax)

    print('Saving data files...')
    ctd_proc = Path('/Users/gunnar/Projects/blt/data/blt1/ctd/proc')
    # Save CTD 24Hz time series both as netcdf and as mat file.
    # c.to_mat(ctd_proc / 'mat' / f'blt_ctd_24hz_{stn:03d}.mat')
    cx.to_netcdf(ctd_proc / 'nc' / f'blt_ctd_24hz_{stn:03d}.nc')
    # Save depth-binned profiles.
    ctd.io.prof_to_mat(ctd_proc / 'mat' / f"blt_ctd_prof_{stn:03d}.mat", datad1m, datau1m)
    datad1m.to_netcdf(ctd_proc / 'nc' / f"blt_ctd_prof_dc_{stn:03d}.nc")
    datau1m.to_netcdf(ctd_proc / 'nc' / f"blt_ctd_prof_uc_{stn:03d}.nc")
    print('Done')


if __name__ == '__main__':
    stn = int(sys.argv[1])
    print(f'Processing cast {stn:03d}')
    proc_raw(stn)
