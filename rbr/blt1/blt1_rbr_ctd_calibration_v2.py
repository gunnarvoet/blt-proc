# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] heading_collapsed=true
# ### Imports

# %% hidden=true
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
from pathlib import Path
import pyrsktools
import os

import gvpy as gv
import rbrmoored as rbr

# %reload_ext autoreload
# %autoreload 2
# %autosave 300
# %config InlineBackend.figure_format = 'retina'

# %% [markdown] hidden=true
# Let's see if we can import the `bltproc` package we created. It will hold some of the processing code to make the notebook less cluttered.

# %% hidden=true
import bltproc as bp
bp.thermistors.test()

# %% [markdown]
# # BLT RBR Solo CTD Calibration v2

# %% [markdown]
# Using the `bltproc` package in this version. Version 1 is still around in `old/` as a legacy piece but not developed any further.
#
# The following happens in this notebook:
# - Extract time period of CTD calibration cast and save thermistor time series for these periods to new files.
# - Determine calibration offsets from CTD casts. Ideally these would be calibration curves, but we do not really have enough calibration data for curves and need to revert to constant offsets.
# - Calibration offsets for the RBRs (excluding the sensors integrated directly into the MAVS packages) are saved to `blt1_rbr_ctd_cal_offsets.nc`.
#
# The following was supposed to happen here but currently lives in the initial processing notebook. Should probably move this here or even better to its own notebook as it depends on both the initial processing and this notebook. Could name that notebook something like level 1.
# - Cut time series for time at depth and save to new file. We may want to hold off with this and do it when we apply the CTD calibration.

# %% [markdown] heading_collapsed=true
# ### Settings

# %% hidden=true
# We are skipping a number of cells. Run them by setting the following to True.
run_all = False

# %% [markdown] hidden=true
# Set paths

# %% hidden=true
# BLT1 moorings
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
# all RBR time series from MAVS1 and MAVS2 are here:
rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo")
# raw data:
data_raw = rbr_dir.joinpath("raw")
# full length processed time series (level 0):
data_out = rbr_dir.joinpath("proc")
# RBR time series at times of CTD calibration casts:
data_out_ctd_cal_cast1 = rbr_dir.joinpath("proc_ctd_cal_cast_20210622")
data_out_ctd_cal_cast2 = rbr_dir.joinpath("proc_ctd_cal_cast_20210623")

# %% [markdown] heading_collapsed=true
# ## Save short time series chunks for development

# %% [markdown] hidden=true
# Saves four days worth of data from each thermistor into `proc_dev` directory for development purposes.

# %% hidden=true
# directory for short chunks of data for development
data_out_proc_dev = rbr_dir.joinpath("proc_dev")
# save chunks
if run_all:
    bp.thermistors.rbr_save_development_chunks(data_out, data_out_proc_dev)

# %% [markdown] heading_collapsed=true
# ## Save CTD cal time series

# %% [markdown] hidden=true
# Extract all thermistor time series for time of calibration cast 002.

# %% hidden=true
ctd_time1 = slice('2021-06-22 07:20:00', '2021-06-22 08:50:00')

# %% hidden=true
if run_all:
    bp.thermistors.rbr_save_ctd_cal_time_series(
        data_out, ctd_time1, data_out_ctd_cal_cast1
    )

# %% [markdown] hidden=true
# Also for the thermistors that went on the second CTD calibration cast (003).

# %% hidden=true
ctd_time2 = slice('2021-06-23 01:30:00', '2021-06-23 03:51:00')

# %% hidden=true
if run_all:
    bp.thermistors.rbr_save_ctd_cal_time_series(
        data_out, ctd_time2, data_out_ctd_cal_cast2
    )

# %% [markdown] heading_collapsed=true
# ## CTD calibration

# %% [markdown] heading_collapsed=true hidden=true
# ### Load CTD data

# %% [markdown] hidden=true
# Load CTD time series. For both cruises, sensor pair 1 was mounted at the bottom of the rosette, sensor pair 2 on the vane.

# %% [markdown] hidden=true
# The first calibration cast (002) went to 1700m (but not close to the bottom) and stopped at 1700m and again around 400m.

# %% hidden=true
ctd1 = xr.open_dataset('/Users/gunnar/Projects/blt/cruises/blt1/data/ctd/proc/nc/blt_ctd_24hz_002.nc')

# %% [markdown] hidden=true
# Some of the thermistors stayed on for the second (deeper) CTD cast (003) that went close to the bottom. Stops at 2798m (bottom), 1500m, and 300m.

# %% hidden=true
ctd2 = xr.open_dataset('/Users/gunnar/Projects/blt/cruises/blt1/data/ctd/proc/nc/blt_ctd_24hz_003.nc')

# %% [markdown] hidden=true
# Both CTD temperature sensors were calibrated in 2020:

# %% hidden=true
print(f'T1: {ctd1.t1.attrs["CalDate"]}')
print(f'T2: {ctd1.t2.attrs["CalDate"]}')

# %% [markdown] hidden=true
# Load all calibration time series for both CTD casts

# %% hidden=true
c1 = bp.thermistors.rbr_load_cals(data_out_ctd_cal_cast1)
c2 = bp.thermistors.rbr_load_cals(data_out_ctd_cal_cast2)

# %% [markdown] hidden=true
# The temperature time series match pretty well overall.

# %% hidden=true
fig, (axa, axb) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                       constrained_layout=True, sharey=True)

c1.plot(hue='n', add_legend=False, ax=axa)
ctd.t1.plot(ax=axa)

for axi, ci, ctdi in zip([axa, axb], [c1, c2], [ctd1, ctd2]):
    ctdi.t1.plot(ax=axi, color='C0')
    ci.plot(hue='n', add_legend=False, ax=axi, color='k', alpha=0.05, linewidth=0.5)

for axi in [axa, axb]:
    axi.set(ylabel='', xlabel='', ylim=(2.8, 10))
    gv.plot.concise_date(axi)
    gv.plot.axstyle(axi, nospine=True, fontsize=11, grid=False)
    axi.set_yticks([])
# axa.set(ylabel='temperature [°C]');

# %% [markdown] hidden=true
# Unfortunately we did not do a lot of stops. The first cast has stops at 1700m and then a couple around 400m.

# %% hidden=true
fig, ax = gv.plot.quickfig()
ctd.p.plot()
timesel1 = [
    slice("2021-06-22 08:03:30", "2021-06-22 08:03:45"),
    slice("2021-06-22 08:04:10", "2021-06-22 08:05:00"),
    slice("2021-06-22 08:28:10", "2021-06-22 08:28:50"),
    slice("2021-06-22 08:31:00", "2021-06-22 08:33:00"),
    slice("2021-06-22 08:35:00", "2021-06-22 08:37:00"),
]
for tsi in timesel1:
    ctd.p.sel(time=tsi).plot(color="r")
ax.set(ylabel='pressure [dbar]', xlabel='', title='BLT1 RBR CTD Calibration Cast (002, 6/22)')
ax.invert_yaxis()
gv.plot.concise_date(ax)
gv.plot.png('blt1_rbr_ctd_cal_cast_20210622')

# %% [markdown] hidden=true
# The second cast has a few more stops.

# %% hidden=true
fig, ax = gv.plot.quickfig()
ctd2.p.plot()
timesel2 = [
    slice("2021-06-23 02:21:30", "2021-06-23 02:23:15"),
    slice("2021-06-23 02:44:00", "2021-06-23 02:46:00"),
    slice("2021-06-23 03:16:40", "2021-06-23 03:17:10"),
    slice("2021-06-23 03:40:10", "2021-06-23 03:40:45"),
]
for tsi in timesel2:
    ctd2.p.sel(time=tsi).plot(color="r")
ax.set(ylabel='pressure [dbar]', xlabel='', title='BLT1 RBR CTD Calibration Cast (003, 6/23)')
ax.invert_yaxis()
gv.plot.concise_date(ax)
gv.plot.png('blt1_rbr_ctd_cal_cast_20210623')

# %% [markdown] heading_collapsed=true hidden=true
# ### Investigate cal stops

# %% [markdown] hidden=true
# How do we go from here? For all stops, plot the time series and refine the slice to pick the most stable periods. Then calculate offsets. We will end up with a number of calibration points for each sensor that (maybe) we can use to fit a curve.

# %% [markdown] hidden=true
# **cast 002**
#
# Messy at the bottom of the cast (which is not near the ocean floor). Highly questionable whether we can use this cast. Maybe by splitting it up into two shorter segments we can get two (somewhat) independent estimates?

# %% hidden=true
ts = slice("2021-06-22 08:02:10", "2021-06-22 08:05:10")
bp.thermistors.plot_cal_stop(ts, c1, ctd1, dt=4)

# %% [markdown] hidden=true
# Maybe use these two periods and compare with sensor 1? Let's use these instead of the long one at the bottom and see by how much they differ. I put them into `timesel1`.

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel1[0], c1, ctd1, dt=4)

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel1[1], c1, ctd1, dt=4)

# %% [markdown] hidden=true
# The following 3 stops are very close together in temperature space

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel1[2], c1, ctd1, dt=4)

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel1[3], c1, ctd1, dt=3)

# %% hidden=true
fig, ax = bp.thermistors.plot_multiple_cal_stop(timesel1, c1, ctd1, dt=4)
fig.suptitle('RBR CTD calibration cast 002')
gv.plot.png('blt1_rbr_ctd_cal_cast_002_stops')

# %% [markdown] hidden=true
# **cast 003**
#
# Not much to get from the stop on the downcast. The other three stops look decent though.

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel2[0], c2, ctd2, dt=4)

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel2[1], c2, ctd2, dt=4)

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel2[2], c2, ctd2, dt=4)

# %% hidden=true
bp.thermistors.plot_cal_stop(timesel2[3], c2, ctd2, dt=4)

# %% hidden=true
fig, ax = bp.thermistors.plot_multiple_cal_stop(timesel2, c2, ctd2, dt=4)
fig.suptitle('RBR CTD calibration cast 003')
gv.plot.png('blt1_rbr_ctd_cal_cast_003_stops')

# %% [markdown] heading_collapsed=true hidden=true
# ### Calculate offsets for one cal stop

# %% [markdown] hidden=true
# Find offsets for all thermistors for one stop. Return offset calculated based on the mean of both CTD sensors and each CTD sensor individually. Return mean temperature of RBR.

# %% [markdown] hidden=true
# We calculate CTD temperature minus thermistor temperature. We then need to add the output to the thermistor time series to calibrate the data towards the CTD values. For example, if the thermistor shows a higher temperature than the CTD, the output will be negative and adding this negative offset to the thermistor data will nudge it towards the colder CTD temperature.

# %% hidden=true
out = bp.thermistors.rbr_ctd_cal_find_offset(timesel1[3], c1, ctd=ctd1)

# %% [markdown] hidden=true
# Okay, the following plot makes sense. For the fourth cal point on the first cast we have more warmer thermistors (compare figure somewhere above), resulting in more negative offsets shown here.

# %% hidden=true
out = out.swap_dims({'m':'sensor'})
out.mean_diff.plot(hue='sn', add_legend=False, color='k', alpha=0.1);

# %% [markdown] hidden=true
# Let's have a look at the standard deviation of the the offset. Somewhere around $5\times10^{-4}$ for most sensors so offsets $\mathcal{0}(10^{-3})$ are statistically significant.

# %% hidden=true
out.std_diff.plot(hue='sn', add_legend=False, color='k', alpha=0.1);

# %% [markdown] hidden=true
# It doesn't really matter which of the two CTD sensors we pick, this makes only a very minor difference.

# %% hidden=true
for g, d in out.groupby('sensor'):
    d.mean_diff.plot.hist(bins=np.arange(-0.01, 0.011, 0.0005), alpha=0.5);

# %% [markdown] heading_collapsed=true hidden=true
# ### Run for all stops cast 002

# %% hidden=true
aa = [bp.thermistors.rbr_ctd_cal_find_offset(ti, c1, ctd=ctd1) for ti in timesel1]

# %% hidden=true
a =xr.concat(aa, dim='calpoint')
a.coords['temp'] = (('calpoint'), a.mean_temp.mean(dim='sn').data)

# %% [markdown] hidden=true
# Really only the bottom stop (cal points 1 and 2) are of use for the mooring deployment as the target temperature range is roughly between 4 and 5. On the other hand, points 1 and 2 are crappy while the shallower points are a little more stable.

# %% hidden=true
print('cal stops at ', a.temp.data)

# %% hidden=true
ab = a.isel(calpoint=[0, 1, 2, 3, 4])

# %% hidden=true
tmp = ab.isel(m=2)
fig, ax = gv.plot.quickfig()
for g, b in tmp.groupby('sn'):
    ax.plot(b.mean_temp, b.mean_diff, 'k-o', alpha=0.2);

# %% [markdown] hidden=true
# Does this give us a reasonable result? It seems like we get an uncertainty of something like 2 millidegrees in this calibration, but some of the offsets are larger than this and thus seem good to be applied in any case.

# %% [markdown] hidden=true
# It may be good to compare the calibration offsets found here for cast with the ones from cast 3, at least for the deep sensors that were calibrated on that cast.

# %% [markdown] hidden=true
# Do we trust one of the cal stops here more than the other? They are both subpar...

# %% [markdown] heading_collapsed=true hidden=true
# ### Run for all stops cast 003

# %% [markdown] hidden=true
# Let's exclude the stop on the downcast.

# %% hidden=true
aa = [bp.thermistors.rbr_ctd_cal_find_offset(ti, c2, ctd=ctd2) for ti in timesel2[1:]]

# %% hidden=true
a =xr.concat(aa, dim='calpoint')
a.coords['temp'] = (('calpoint'), a.mean_temp.mean(dim='sn').data)

# %% hidden=true
print('cal stops at ', a.temp.data)

# %% [markdown] hidden=true
# Only the first two points should be used, or really, maybe only the second as the target temperature range is between 4 and 5 degrees. However, offsets seem to be pretty stable between the second and third stop. Maybe this indicates that working with just a constant offset isn't too bad after all and could just be the best we can do with this sub-optimal calibration dataset.

# %% hidden=true
tmp = a.isel(m=2)
fig, ax = gv.plot.quickfig()
for g, b in tmp.groupby('sn'):
    ax.plot(b.mean_temp, b.mean_diff, 'k-x', alpha=0.2);

# %% [markdown] heading_collapsed=true hidden=true
# ### Combine results from both calibration casts

# %% [markdown] hidden=true
# Combine cast 003 and cast 002:

# %% hidden=true
c = xr.concat([a, ab], dim='calpoint')

# %% [markdown] hidden=true
# This gives us 8 cal stops total. 

# %% hidden=true
c.temp.data

# %% [markdown] hidden=true
# We want to exclude the two bottom stops on cast 002 as they seem to be too messy to give us reliable results:

# %% hidden=true
selected_points = [0, 1, 2, 5, 6, 7]

# %% [markdown] hidden=true
# Plot calibration offsets at all cal stops for each sensor. This will generate individual figures in `cal_offsets/`.

# %% hidden=true
if run_all:
    bp.thermistors.rbr_ctd_cal_plot_all_sensor_offsets(c, selected_points)

# %% [markdown] hidden=true
# Calculate offsets from the cal stops selected above for sensor 1 and the combination of sensor 1 & 2 and then compare results. If they don't differ too much, let's go with the combination as the thermistors were mounted a little bit all over the place on the rosette.

# %% hidden=true
offsets = c.isel(calpoint=selected_points).mean_diff.mean(dim='calpoint')

# %% hidden=true
fig, ax = gv.plot.quickfig()
ax.hist(offsets.isel(m=0).data, bins=np.arange(-1e-2, 1.1e-2, 1e-3));

# %% [markdown] hidden=true
# The difference between offsets from sensor 1 and from the combination of sensors 1 & 2 is negligible:

# %% hidden=true
fig, ax = gv.plot.quickfig()
ax.plot(offsets.isel(m=0).data)
ax.plot(offsets.isel(m=2).data);

# %% [markdown] heading_collapsed=true hidden=true
# ### Save calibration offsets

# %% [markdown] hidden=true
# Pick the result from the combination of sensors and save.

# %% hidden=true
out = offsets.isel(m=2).drop('sensor').squeeze()
out.name = 'offset'

# %% hidden=true
out.attrs['info'] = 'offsets here were calculated as CTD minus thermistor and thus have to be added to the thermistor time series to calibrate it towards the CTD measurements.'

# %% hidden=true
out.to_netcdf('blt1_rbr_ctd_cal_offsets.nc', mode='w')

# %% [markdown] heading_collapsed=true
# ## why we can't do more than constant offset

# %% [markdown] hidden=true
# Just one example showing that a second degree polynomial fit to all calibration points does not make sense. The outliers dominate the fit and should rather be excluded. Once that's done, a constant offset wouldn't differ too much from a linear fit to the offsets.

# %% hidden=true
example_sensor = c.isel(sn=20, m=2)

# %% hidden=true
y = example_sensor.mean_diff
x = example_sensor.mean_temp

pf = np.polynomial.polynomial.polyfit(x, y, deg=2)
x2 = np.arange(0, 12)
py = np.polynomial.polynomial.polyval(x2, pf)

# %% hidden=true
fig, ax = gv.plot.quickfig()
ax.plot(x2, py*1e3, color='r', linewidth=0.5)
ax.plot(x, (y * 1e3), "kx", alpha=0.8)
ax.set(
    ylim=(-5, 5),
    title=f"BLT1 RBR {sn} CTD Cal Offsets",
    xlabel="temperature [°C]",
    ylabel="temperature offset [mK]",
);

# %% [markdown] hidden=true
# A linear fit is definitely better than a second degree polynomial, but even then a constant offset may introduce less uncertainty or error than a fit.

# %% hidden=true
y = example_sensor.mean_diff
x = example_sensor.mean_temp

pf = np.polynomial.polynomial.polyfit(x, y, deg=1)
x2 = np.arange(0, 12)
py = np.polynomial.polynomial.polyval(x2, pf)

# %% hidden=true
fig, ax = gv.plot.quickfig()
ax.plot(x2, py*1e3, color='r', linewidth=0.5)
ax.plot(x, (y * 1e3), "kx", alpha=0.8)
ax.set(
    ylim=(-5, 5),
    title=f"BLT1 RBR {sn} CTD Cal Offsets",
    xlabel="temperature [°C]",
    ylabel="temperature offset [mK]",
);
