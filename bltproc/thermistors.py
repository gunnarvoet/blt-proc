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
    files = sorted(l0_data.glob("*.nc"))

    def save_ctd_cal(file, ctd_time):
        r = xr.open_dataarray(file)
        datestr = ctd_time.start[:10].replace("-", "")
        outname = f'{file.stem[:file.stem.find("_")]}_ctd_cal_cast_{datestr}.nc'
        outpath = save_dir.joinpath(outname)
        r.sel(time=ctd_time).to_netcdf(outpath)
        r.close()

    for file in files:
        save_ctd_cal(file, ctd_time)


def rbr_load_cals(cal_dir):
    calfiles = sorted(cal_dir.glob("*.nc"))
    cals = [xr.open_dataarray(calfile) for calfile in calfiles]
    sns = [ci.attrs["SN"] for ci in cals]

    # pick one time for all of them and interpolate
    time = cals[0].time.copy()
    calsi = [ci.interp_like(time) for ci in cals]
    c = xr.concat(calsi, dim="n")
    c["sn"] = (("n"), sns)
    return c


def plot_zoom(ax, ts, rbr, ctd, add_legend=False):
    t = rbr.sel(time=ts)
    t.plot(
        hue="n", add_legend=False, linewidth=0.75, color="k", alpha=0.3, ax=ax
    )
    ctd.t1.sel(time=ts).plot(color="C0", linewidth=1, label="CTD 1", ax=ax)
    ctd.t2.sel(time=ts).plot(color="C4", linewidth=1, label="CTD 2", ax=ax)
    ax.set(xlabel="", ylabel="in-situ temperature [°C]")
    gv.plot.concise_date(ax)
    gv.plot.axstyle(ax, fontsize=9)
    if add_legend:
        ax.legend()


def plot_cal_stop(ts, rbr, ctd, dt=2):
    """Plot CTD calibration stop with zoom."""
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True
    )
    plot_zoom(ax[0], ts, rbr, ctd, add_legend=True)
    tsn = slice(
        np.datetime64(ts.start) - np.timedelta64(dt, "m"),
        np.datetime64(ts.stop) + np.timedelta64(dt, "m"),
    )
    plot_zoom(ax[1], tsn, rbr, ctd)
    mctd = ctd.t1.sel(time=ts).mean()
    ax[1].plot(
        [np.datetime64(ti) for ti in [ts.start, ts.stop]],
        np.tile(mctd - 0.02, 2),
        "r",
    )


def plot_multiple_cal_stop(ts, rbr, ctd, dt=2):
    """Plot multiple CTD calibration stop with zoom."""
    n = len(ts)
    fig, ax = plt.subplots(
        nrows=n, ncols=2, figsize=(10, 3*n), constrained_layout=True,
    )
    leg = True
    ii = 1
    for tsi, axi in zip(ts, ax):
        plot_zoom(axi[0], tsi, rbr, ctd, add_legend=leg)
        tsn = slice(
            np.datetime64(tsi.start) - np.timedelta64(dt, "m"),
            np.datetime64(tsi.stop) + np.timedelta64(dt, "m"),
        )
        gv.plot.annotate_corner(ii, axi[0], background_circle='0.1', col='w')
        plot_zoom(axi[1], tsn, rbr, ctd)
        mctd = ctd.t1.sel(time=tsi).mean()
        axi[1].plot(
            [np.datetime64(ti) for ti in [tsi.start, tsi.stop]],
            np.tile(mctd - 0.02, 2),
            "r",
        )
        leg=False
        ii += 1
    return fig, ax


def rbr_ctd_cal_find_offset(ts, rbr, ctd):
    ctdm1 = ctd.t1.sel(time=ts).mean().data
    ctdm2 = ctd.t2.sel(time=ts).mean().data
    print(f'Difference between CTD sensor 1 & 2 mean values: {ctdm1-ctdm2:.4e}°C')
    ctdm = np.mean([ctdm1, ctdm2])
    ab = rbr.sel(time=ts)
    diffmean_1 = (ctdm1-ab).mean(dim='time')
    diffmean_1.name = 'mean1'
    diffstd_1 = (ctdm1-ab).std(dim='time')
    diffstd_1.name = 'std1'
    diffmean_2 = (ctdm2-ab).mean(dim='time')
    diffmean_2.name = 'mean2'
    diffstd_2 = (ctdm2-ab).std(dim='time')
    diffstd_2.name = 'std2'
    diffmean_both = (ctdm-ab).mean(dim='time')
    diffmean_both.name = 'meanboth'
    diffstd_both = (ctdm-ab).std(dim='time')
    diffstd_both.name = 'stdboth'
    mean = xr.concat([diffmean_both, diffmean_1, diffmean_2], dim='m')
    mean = mean.swap_dims({'n':'sn'})
    mean.name = 'mean_diff'
    std = xr.concat([diffstd_both, diffstd_1, diffstd_2], dim='m')
    std = std.swap_dims({'n':'sn'})
    std.name = 'std_diff'

    mean_temp = ab.mean(dim='time')
    mean_temp.name = 'mean_temp'
    mean_temp = mean_temp.swap_dims({'n':'sn'})

    out = xr.merge([mean, std, mean_temp])
    out.coords['sensor'] = (('m'), ['1', '2', 'both'])

    return out


def rbr_ctd_cal_plot_all_sensor_offsets(c, selected_points):
    tmp = c.isel(m=2)
    # tmp = tmp.isel(sn=0)
    fig, ax = gv.plot.quickfig()
    for sn, b in tmp.groupby('sn'):
        gv.plot.axstyle(ax)

        ax.hlines(b.isel(calpoint=selected_points).mean_diff.mean() * 1e3, 0, 11, color='b', linewidth=0.7)
        ax.plot(b.mean_temp[:3], (b.mean_diff[:3] * 1e3), "k1", alpha=0.8)
        ax.plot(b.mean_temp[3], (b.mean_diff[3] * 1e3), color="k", marker=6, linestyle='', alpha=0.8)
        ax.plot(b.mean_temp[4], (b.mean_diff[4] * 1e3), color="k", marker=7, linestyle='', alpha=0.8)
        ax.plot(b.mean_temp[5:], (b.mean_diff[5:] * 1e3), "k2", alpha=0.8)
        ax.set(
            ylim=(-10, 10),
            xlim=(2, 11),
            title=f"BLT1 RBR {sn} CTD Cal Offsets",
            xlabel="temperature [°C]",
            ylabel="temperature offset [mK]",
        )
        if sn == 72216:
            ax.set(ylim=(-20, 20))
        gv.plot.png(fname=f'{sn:06}_cal_offsets', figdir='cal_offsets', verbose=False)
        ax.cla()
