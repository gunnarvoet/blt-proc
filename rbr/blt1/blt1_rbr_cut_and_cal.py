# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: python3 (blt-proc)
#     language: python
#     name: conda-env-blt-proc-py
# ---

# %% [markdown]
# ### Imports

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import pyrsktools
import os

import gvpy as gv
import rbrmoored as rbr

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
import bltproc as bp
bp.thermistors.test()

# %%
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # BLT RBR Solo Cut & Cal

# %% [markdown]
# - Cut RBR themistor time series to time at depth and before data start to drop out for some sensors.
# - Apply offsets determined in CTD calibration casts.
# - Save to level 1 data directory.
# - Plot each time series for visual inspection.

# %% [markdown]
# Most of the code moved the `bltproc.thermistors`.

# %% [markdown]
# ### Set processing parameters

# %%
# We are skipping a number of cells. Run them by setting the following to True.
run_all = False

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo")
level0_dir = rbr_dir.joinpath("proc")
# directory for level 1 processed data
level1_dir = rbr_dir.joinpath("proc_L1")

# %%
print(level1_dir)

# %% [markdown]
# ## Cut and save calibrated time series

# %% [markdown]
# We want to save a level 1 dataset just as we did for the SBE56. This means determining the end of the good part of the time series, either where it stops or where it starts showing gaps due to the battery dropping out, and to apply the CTD calibration.

# %%
# read a test unit
sn = 207286
tmp = bp.thermistors.rbr_load_proc_level0(sn, level0_dir)

# %%
# read a test unit
sn = 207383
tmp = bp.thermistors.rbr_load_proc_level0(sn, level0_dir)

# %%
# read a test unit
sn = 72188
tmp = bp.thermistors.rbr_load_proc_level0(sn, level0_dir)

# %%
attrs = tmp.attrs

# %%
attrs

# %% [markdown]
# Find end times

# %%
last_time = bp.thermistors.rbr_find_last_time_stamp(tmp)
print(last_time)

# %%
tdi = bp.thermistors.rbr_find_gaps(tmp)
if len(tdi) > 0:
    print(tdi.time.data-tdi.data)
    print([np.timedelta64(t, 's') for t in tdi.data])

# %%
print([np.timedelta64(t, 'h') for t in tdi.data])

# %%
ti = [np.timedelta64(t, 'h') for t in tdi.data] > np.timedelta64(1, 'h')

# %%
ti = tdi > np.timedelta64(1, 'h')

# %%
t = tdi.where(ti, drop=True)

# %%
t0 = t.isel(time=0)
t0['time'] = (t0.time-t0).data

# %%
t0

# %% [markdown]
# Like with the SBE56, we stop the time series where gaps become longer than one hour.

# %%
t = bp.thermistors.rbr_find_first_long_gap(tdi)

# %%
if np.isnat(t):
    print('no log gap')
else:
    t.time.data

# %% [markdown]
# Load CTD calibration offsets.

# %%
ctdcal = xr.open_dataarray('/Users/gunnar/Projects/blt/proc/rbr/blt1/blt1_rbr_ctd_cal_offsets.nc')

# %%
ctdcal.sel(sn=sn)

# %% [markdown]
# Apply calibration offset to the test dataset.

# %%
tmpc = bp.thermistors.rbr_apply_ctd_offset(thermistor=tmp, sn=sn, ctdcal=ctdcal)

# %%
tmp.sel(time='2021-08-01 01:00').plot()
tmpc.sel(time='2021-08-01 01:00').plot()

# %% [markdown]
# Load mooring sensor info to know which sensor on which mooring.

# %%
path_to_thermistor_info = Path('/Users/gunnar/Projects/blt/moorings/thermistors/')

# %%
mavs1_rbr, mavs2_rbr = bp.thermistors.rbr_blt1_load_mooring_sensor_info(path_to_thermistor_info)

# %%
mavs1_rbr.head()

# %% [markdown]
# We also need to cut at least one time series short manually. The calibration procedure showed that 72188 starts to develop a lot of drift starting August 5th.

# %%
end_manually = dict()
end_manually[72188] = np.datetime64('2021-08-05 00:00:00')

# %% [markdown]
# Test running the cut & cal function.

# %%
test = bp.thermistors.rbr_cut_and_cal(
    sn, level0_dir, level1_dir, path_to_thermistor_info, ctdcal, end_manually
)

# %%
test.plot();

# %% [markdown]
# Run for all RBRs and save to level 1 directory.

# %%
if run_all:
    sn_mavs1 = mavs1_rbr.index.to_numpy()
    for sni in sn_mavs1:
        test = bp.thermistors.rbr_cut_and_cal(
            sni, level0_dir, level1_dir, path_to_thermistor_info, ctdcal
        )

    sn_mavs2 = mavs2_rbr.index.to_numpy()
    for sni in sn_mavs2:
        test = bp.thermistors.rbr_cut_and_cal(
            sni, level0_dir, level1_dir, path_to_thermistor_info, ctdcal
        )

# %% [markdown]
# ## Plot all L1 time series for inspection

# %%
fig_dir = Path('inspect_level1')

# %%
l1_files = sorted(level1_dir.glob('*.nc'))


# %%
def plot_thermistor_time_series(t, ax):
    gv.plot.axstyle(ax)
    t.plot(ax=ax, linewidth=0.5)
    gv.plot.concise_date(ax)
    sn = t.attrs['SN'] if 'SN' in t.attrs else t.attrs['sn']
    ax.set(xlabel='', ylabel='temperature [$^{\circ}$C]', title=sn)
    plotname = f'{sn:06}_level1'
    gv.plot.png(plotname, figdir=fig_dir, verbose=False)
    ax.cla()


# %%
if run_all:
    fig, ax = gv.plot.quickfig()
    for file in l1_files:
        tmp = xr.open_dataarray(file)
        plot_thermistor_time_series(tmp, ax)
