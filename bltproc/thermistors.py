#!/usr/bin/env python
"""Module for processing moored BLT thermistors."""

from pathlib import Path
import os
import gvpy as gv
import numpy as np
import xarray as xr
import pandas as pd
import gsw
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def test():
    print("hello world!")


def rbr_save_development_chunks(l0_data, chunk_dir):
    # save short chunks of data for development
    time_span = slice("2021-07-31 00:00:00", "2021-08-04 00:00:00")
    files = sorted(l0_data.glob("*.nc"))

    def save_short_chunk(file, ctd_time):
        r = xr.open_dataarray(file)
        outname = f'{file.stem[:file.stem.find("_")]}_proc_dev.nc'
        outpath = chunk_dir.joinpath(outname)
        r.sel(time=ctd_time).to_netcdf(outpath)
        r.close()

    for file in files:
        save_short_chunk(file, time_span)


def rbr_save_ctd_cal_time_series(l0_data, ctd_time, save_dir):
    """Save thermistor time series for time of CTD calibration cast."""
    files = sorted(l0_data.glob('*.nc'))

    def save_ctd_cal(file, ctd_time):
        r = xr.open_dataarray(file)
        datestr = ctd_time.start[:10].replace('-', '')
        outname = f'{file.stem[:file.stem.find("_")]}_ctd_cal_cast_{datestr}.nc'
        outpath = save_dir.joinpath(outname)
        r.sel(time=ctd_time).to_netcdf(outpath)
        r.close()

    for file in files:
        save_ctd_cal(file, ctd_time)


def rbr_load_cals(cal_dir):
    calfiles = sorted(cal_dir.glob('*.nc'))
    cals = [xr.open_dataarray(calfile) for calfile in calfiles]
    sns = [ci.attrs['SN'] for ci in cals]

    # pick one time for all of them and interpolate
    time = cals[0].time.copy()
    calsi = [ci.interp_like(time) for ci in cals]
    c = xr.concat(calsi, dim='n')
    c['sn'] = (('n'), sns)
    return c


def plot_zoom(ax, ts, rbr, ctd):
    t = rbr.sel(time=ts)
    t.plot(hue="n", add_legend=False, linewidth=0.75, color="k", alpha=0.3, ax=ax)
    ctd.t1.sel(time=ts).plot(color="C0", linewidth=1, label="CTD 1", ax=ax)
    ctd.t2.sel(time=ts).plot(color="C4", linewidth=1, label="CTD 2", ax=ax)
    ax.set(xlabel="", ylabel="in-situ temperature [°C]")
    gv.plot.concise_date(ax)
    gv.plot.axstyle(ax, fontsize=9)
    ax.legend()


def plot_cal_stop(ts, rbr, ctd, dt=2):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
    plot_zoom(ax[0], ts, rbr, ctd)
    tsn = slice(
        np.datetime64(ts.start) - np.timedelta64(dt, "m"),
        np.datetime64(ts.stop) + np.timedelta64(dt, "m"),
    )
    plot_zoom(ax[1], tsn, rbr, ctd)
    mctd = ctd.t1.sel(time=ts).mean()
    ax[1].plot([np.datetime64(ti) for ti in [ts.start, ts.stop]], np.tile(mctd-0.02, 2), 'r')
