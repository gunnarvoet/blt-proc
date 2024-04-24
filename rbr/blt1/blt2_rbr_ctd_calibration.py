# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python [conda env:blt-proc]
#     language: python
#     name: conda-env-blt-proc-py
# ---

# %% [markdown]
# ### Imports

# %%
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import xarray as xr

import gvpy as gv
import rbrmoored as rbr

# %reload_ext autoreload
# %autoreload 2
# %autosave 300
# %config InlineBackend.figure_format = 'retina'

# %%
import bltproc as bp
bp.thermistors.test()

# %%
logger = gv.misc.log()

logger.info("Working on MAVS3 & MAVS4 thermistor calibration")

# %% [markdown]
# # BLT MAVS 3 & 4 RBR Solo CTD Calibration

# %% [markdown]
# Using the `bltproc` package in this version.
#
# The following happens in this notebook:
# - Extract time period of CTD calibration cast and save thermistor time series for these periods to new files.
# - Determine calibration offsets from CTD casts. Ideally these would be calibration curves, but we do not really have enough calibration data for curves and need to revert to constant offsets.
# - Calibration offsets for the RBRs (excluding the sensors integrated directly into the MAVS packages) are saved to `blt2_rbr_ctd_cal_offsets.nc`.

# %% [markdown] heading_collapsed=true
# ### Settings

# %% hidden=true
# We are skipping a number of cells. Run them by setting the following to True.
run_all = False

# %% [markdown] hidden=true
# Set paths

# %% hidden=true
# BLT moorings
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
# all RBR time series from MAVS3 and MAVS4 are here:
rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo_2")
# raw data:
data_raw = rbr_dir.joinpath("raw")
# full length processed time series (level 0):
data_out = rbr_dir.joinpath("proc")
# RBR Solo time series at times of CTD calibration casts:
data_out_ctd_cal_cast = rbr_dir.joinpath("proc_ctd_cal_cast_202110")
# CTD directory
ctd_dir = Path("/Users/gunnar/Projects/blt/data/blt2/ctd/proc/nc")

# %% [markdown]
# ### Save RBR Solo data for time of CTD calibration

# %% [markdown]
# Extract all RBR time series for time of CTD rosette calibration casts. The calibration happened either on Oct 13 16:00 or Oct 15 15:00 so we just extract the time that covers both of these casts.

# %%
tmp2 = xr.open_dataarray('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc_ctd_cal_cast_202110/102981_ctd_cal_cast_20211013.nc')

# %%
tmp2.sel(time='2021-10-13').gv.plot()

# %%
tmp2 = xr.open_dataarray('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc/102974_20220809_1800.nc')

# %%
tmp2.time[0]

# %%
tmp2.gv.plot()

# %%
ctd_time = slice('2021-10-13 00:00:00', '2021-10-16 00:00:00')

# %%
ctd_time.start

# %%
run_all = True

# %%
if run_all:
    bp.thermistors.rbr_save_ctd_cal_time_series(
        data_out, ctd_time, data_out_ctd_cal_cast
    )

# %% [markdown]
# ### Load CTD data

# %% [markdown]
# Load CTD time series. For both cruises, sensor pair 1 was mounted at the bottom of the rosette, sensor pair 2 on the vane.

# %% [markdown]
# Cast 62 was `Oct 13 2021 16:37:38`. 
# Cast 72 was `Oct 15 2021 15:11:06`.

# %%
ctd1 = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/blt2/ctd/proc/nc/blt_ctd_24hz_062.nc"
)
ctd1_prof = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/blt2/ctd/proc/nc/blt_ctd_prof_uc_062.nc"
)

# %%
ctd2 = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/blt2/ctd/proc/nc/blt_ctd_24hz_072.nc"
)
ctd2_prof = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/blt2/ctd/proc/nc/blt_ctd_prof_uc_072.nc"
)

# %% [markdown]
# Plot both pressure time series

# %%
fig, ax = gv.plot.quickfig(grid=True, h=3, w=5)
ctd1.p.gv.plot(ax=ax)
ctd2.p.gv.plot(ax=ax)
gv.plot.ydecrease()
ht = ax.set(ylabel="pressure [dbar]")

# %% [markdown]
# Both CTD temperature sensors were calibrated in 2020:

# %%
print(f'T1: {ctd1.t1.attrs["CalDate"]}')
print(f'T2: {ctd1.t2.attrs["CalDate"]}')

# %% [markdown]
# Load all calibration time series for both CTD casts

# %%
c_all = bp.thermistors.rbr_load_cals(data_out_ctd_cal_cast)

# %%
c_all.resample(time="6000s").mean().plot()

# %% [markdown]
# Separate between the two CTD casts.

# %%
c1 = (c_all
      .where(c_all.sel(time="2021-10-13 17:15:00", method="nearest") < 10)
      .dropna(dim="n", how="all")
     )
c2 = (c_all
      .where(c_all.sel(time="2021-10-13 17:15:00", method="nearest") > 10)
      .dropna(dim="n", how="all")
     )

# %% [markdown]
# Make sure this worked as intended.

# %%
fig, ax = gv.plot.quickfig(r=2, sharex=True, grid=True)
c1.resample(time="6000s").mean().plot(ax=ax[0])
c2.resample(time="6000s").mean().plot(ax=ax[1])
for axi in ax:
    axi.set(xlabel="")


# %% [markdown]
# ### Auto pick cal stops

# %% [markdown]
# We want to find start and end times of calibration stops. Also, determine good stops via their vertical temperature gradients (less is better for a more stable reference temperature).
#
# First, determine calibration stops automatically. Then manually pick the best ones best on looks. Temperature gradients at stops help narrow down the search for good stops.

# %%
def find_cal_stops(ctd, dp_threshold=0.15e-9):
    time_at_max_p = ctd.time.isel(time=ctd.p.argmax()).data
    p = ctd.p.where(ctd.time >= time_at_max_p, drop=True)
    p = p.resample(time="5s").mean()
    isstop = p.copy() * 0

    stop_i = []
    stop = 0
    prior_was_stop = False
    for ti, pi in p.differentiate(coord="time").groupby("time"):
        if pi > -dp_threshold and not prior_was_stop:
            stop += 1
            prior_was_stop = True
            stop_i.append(stop)
        elif pi > -dp_threshold and prior_was_stop:
            stop_i.append(stop)
        elif pi <= -dp_threshold:
            stop_i.append(-1)
            prior_was_stop = False
    max_stop = stop
    logger.info(f"number of cal stops found: {max_stop}")

    stop_i_da = xr.DataArray(stop_i, coords=dict(time=p.time))

    tmin, tmax = [], []
    for i in range(1, max_stop):
        pp = p.time.where(stop_i_da == i, drop=True)
        tmin.append(pp.time.min(dim="time").data)
        tmax.append(pp.time.max(dim="time").data)
    tmin = np.array(tmin)
    tmax = np.array(tmax)

    timesel = [slice(tmini, tmaxi) for tmini, tmaxi in zip(tmin, tmax)]

    # Drop stops that are zero length
    time_at_stop_in_s = np.array(
        [gv.time.timedelta64_to_s(ts.stop - ts.start) for ts in timesel]
    )
    timesel = [tsi if dt > 0 else None for tsi, dt in zip(timesel, time_at_stop_in_s)]
    timesel = [tsi for tsi in timesel if tsi is not None]
    
    # time between stops
    time_between_stops_in_s = []
    n = len(timesel)
    for i in range(n-1):
        dt = timesel[i+1].start - timesel[i].stop
        time_between_stops_in_s.append(gv.time.timedelta64_to_s(dt))

    # combine stops with very little time in between?
    
    return timesel


# %%
def temp_grad_at_stop(timesel, ctd_prof):
    """Calculate vertical temperature gradient during stop.
    """
    # Temperature gradient
    delta_t = ctd_prof.t1.rolling(depth=5).mean().differentiate(coord="depth")
    # Temperature gradient vs time
    delta_t_time = delta_t.swap_dims(depth="time")
    delta_t_time = delta_t_time.where(~np.isnat(delta_t_time.time), drop=True)
    delta_t_time = delta_t_time.sortby("time")
    # Temperature gradient during stop
    tgrad_at_stop = []
    for ts in timesel:
        if type(ts) is slice:
            d = delta_t_time.sel(time=ts)
            if len(d) > 0:
                tgrad_at_stop.append(d.mean().item())
            else:
                tgrad_at_stop.append(np.nan)

    fig, ax = gv.plot.quickfig(grid=True)
    ax.plot(tgrad_at_stop, "ko")
    t = ax.set(xlabel="stop #", ylabel="temperature gradient [K/m]")


# %% [markdown]
# #### First Calibration Cast

# %%
# p = ctd1.p.sel(time=slice("2021-10-13 17:15:00", "2021-10-13 18:23:00"))
timesel1_auto = find_cal_stops(ctd1, dp_threshold=0.1e-9)

# %%
time_at_stop_in_s = np.array([gv.time.timedelta64_to_s(ts.stop - ts.start) for ts in timesel1_auto ])
print(time_at_stop_in_s)

# %% [markdown]
# Average temperature gradient per stop. Looks like there is a pretty well mixed bottom boundary layer for the first two stops that we will make use of. One of them is also the one where we hung out for about five minutes (compared to just about one to two minutes for the other stops).

# %%
temp_grad_at_stop(timesel1_auto , ctd1_prof)

# %%
fig, ax = gv.plot.quickfig()
ctd1.p.plot()
for tsi in timesel1_auto :
    ctd1.p.sel(time=tsi).plot(color="r")
ax.set(ylabel='pressure [dbar]', xlabel='', title='BLT2 RBR CTD Calibration Cast (062, 2021-10-13)')
ax.invert_yaxis()
gv.plot.concise_date(ax)
# gv.plot.png('blt2_rbr_ctd_cal_cast_20211013')

# %% [markdown]
# #### Second Calibration Cast

# %%
# p = ctd1.p.sel(time=slice("2021-10-13 17:15:00", "2021-10-13 18:23:00"))
timesel2_auto  = find_cal_stops(ctd2, dp_threshold=0.1e-9)

# %%
time_at_stop_in_s = np.array([gv.time.timedelta64_to_s(ts.stop - ts.start) for ts in timesel2_auto ])
print(time_at_stop_in_s)

# %%
fig, ax = gv.plot.quickfig()
ctd2.p.plot()
for tsi in timesel2_auto :
    ctd2.p.sel(time=tsi).plot(color="r")
ax.set(ylabel='pressure [dbar]', xlabel='', title='BLT2 RBR CTD Calibration Cast (072, 2021-10-13)')
ax.invert_yaxis()
gv.plot.concise_date(ax)
# gv.plot.png('blt2_rbr_ctd_cal_cast_20211013')

# %%
temp_grad_at_stop(timesel2_auto , ctd2_prof)

# %% [markdown]
# ### Manually pick cal stops

# %% [markdown]
# Note: Temperature sensor 1 is much more stable for both casts so we'll be using that for calibration.

# %% [markdown]
# #### First cast

# %%
timesel1 = [
    slice("2021-10-13 17:19:00", "2021-10-13 17:24:30"),
    slice("2021-10-13 17:47:30", "2021-10-13 17:48:45"),
    slice("2021-10-13 17:54:30", "2021-10-13 17:55:25"),
    slice("2021-10-13 18:22:00", "2021-10-13 18:22:50"),
]

# %%
ts = timesel1[0]
bp.thermistors.plot_cal_stop(ts, c1, ctd1, dt=4)

# %%
fig, ax = bp.thermistors.plot_multiple_cal_stop(timesel1, c1, ctd1, dt=4)
fig.suptitle('BLT2 RBR CTD calibration cast 062')
gv.plot.png('blt2_rbr_ctd_cal_cast_062_stops')

# %% [markdown]
# #### Second cast

# %%
timesel2 = [
    slice("2021-10-15 16:27:00", "2021-10-15 16:29:00"),
    slice("2021-10-15 16:35:15", "2021-10-15 16:35:55"),
    slice("2021-10-15 16:39:20", "2021-10-15 16:40:10"),
    slice("2021-10-15 16:52:10", "2021-10-15 16:53:10"),
]

# %%
ts = timesel2[0]
bp.thermistors.plot_cal_stop(ts, c2, ctd2, dt=4)

# %%
fig, ax = bp.thermistors.plot_multiple_cal_stop(timesel2, c2, ctd2, dt=4)
fig.suptitle('BLT2 RBR CTD calibration cast 072')
gv.plot.png('blt2_rbr_ctd_cal_cast_072_stops')

# %% [markdown]
# ### Calculate offsets for one cal stop

# %% [markdown]
# Find offsets for all thermistors for one stop. Return offset calculated based on the mean of both CTD sensors and each CTD sensor individually. Return mean temperature of RBR.

# %% [markdown]
# We calculate CTD temperature minus thermistor temperature. We then need to add the output to the thermistor time series to calibrate the data towards the CTD values. For example, if the thermistor shows a higher temperature than the CTD, the output will be negative and adding this negative offset to the thermistor data will nudge it towards the colder CTD temperature.

# %%
out = bp.thermistors.rbr_ctd_cal_find_offset(timesel2[0], c2, ctd=ctd2)

# %%
out = out.swap_dims({'m':'sensor'})
out.mean_diff.plot(hue='sn', add_legend=False, color='k', alpha=0.1);

# %%
out.std_diff.plot(hue='sn', add_legend=False, color='k', alpha=0.1);

# %%
for g, d in out.groupby('sensor'):
    d.mean_diff.plot.hist(bins=np.arange(-0.01, 0.011, 0.0005), alpha=0.5);

# %% [markdown]
# ### Run for all stops cast 062

# %%
aa = [bp.thermistors.rbr_ctd_cal_find_offset(ti, c1, ctd=ctd1) for ti in timesel1]

# %%
a =xr.concat(aa, dim='calpoint')
a.coords['temp'] = (('calpoint'), a.mean_temp.mean(dim='sn').data)

# %% [markdown]
# Really only the bottom stop (cal points 1 and 2) are of use for the mooring deployment as the target temperature range is roughly between 4 and 5.

# %%
print('cal stops at ', a.temp.data)

# %%
ab = a.isel(calpoint=[0, 1, 2, 3, ])

# %%
tmp = ab.isel(m=2)
fig, ax = gv.plot.quickfig()
for g, b in tmp.groupby('sn'):
    ax.plot(b.mean_temp, b.mean_diff, 'k-o', alpha=0.2);

# %% [markdown]
# The second cal stop doesn't really settle to a constant temperature, let's exclude it.

# %%
ab1 = a.isel(calpoint=[0, 2, 3, ])

# %%
tmp = ab1.isel(m=0)
fig, ax = gv.plot.quickfig()
for g, b in tmp.groupby('sn'):
    ax.plot(b.mean_temp, b.mean_diff, 'k-o', alpha=0.2);

# %% [markdown]
# ### Run for all stops cast 072

# %%
aa = [bp.thermistors.rbr_ctd_cal_find_offset(ti, c2, ctd=ctd2) for ti in timesel2]

# %%
a =xr.concat(aa, dim='calpoint')
a.coords['temp'] = (('calpoint'), a.mean_temp.mean(dim='sn').data)

# %%
print('cal stops at ', a.temp.data)

# %%
ab2 = a.isel(calpoint=[0, 1, 2, 3, ])

# %%
tmp = ab2.isel(m=0)
fig, ax = gv.plot.quickfig()
for g, b in tmp.groupby('sn'):
    ax.plot(b.mean_temp, b.mean_diff, 'k-o', alpha=0.2);

# %% [markdown]
# ### Combine results from both calibration casts

# %%
mab1 = ab1.isel(m=0).mean(dim="calpoint")
mab2 = ab2.isel(m=0).mean(dim="calpoint")

# %%
c = xr.concat([mab1, mab2], dim='sn')

# %%
fig, ax = gv.plot.quickfig()
ax.plot(c.mean_diff.data, "ko");

# %%
offsets = c.mean_diff

# %%
offsets.sn.sortby(offsets.sn)

# %% [markdown]
# ### Save calibration offsets

# %%
out = offsets.drop('sensor').squeeze()
out.name = 'offset'

out = out.sortby("sn")

out.attrs['info'] = 'offsets here were calculated as CTD minus thermistor and thus have to be added to the thermistor time series to calibrate it towards the CTD measurements.'

# %%
out.to_netcdf('blt2_rbr_ctd_cal_offsets.nc', mode='w')
