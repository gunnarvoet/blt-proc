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

# %%
logger = gv.misc.log()
logger.info("Working on MAVS3 & MAVS4 RBR thermistors")

# %% [markdown]
# # BLT2 RBR Solo Cut & Cal

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
# BLT moorings
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
# all RBR time series from MAVS3 and MAVS4 are here:
rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo_2")
level0_dir = rbr_dir.joinpath("proc")
# directory for level 1 processed data
level1_dir = rbr_dir.joinpath("proc_L1")

# %%
print(level0_dir)
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

# %% [markdown]
# For the BLT2 deployment pretty much all RBRs worked as expected.

# %%
tdi = bp.thermistors.rbr_find_gaps(tmp)
if len(tdi) > 0:
    print(tdi.time.data-tdi.data)
    print([np.timedelta64(t, 's') for t in tdi.data])

# %%
print([np.timedelta64(t, 'h') for t in tdi.data])

# %% [markdown]
# Load CTD calibration offsets.

# %%
ctdcal = xr.open_dataarray(
    "/Users/gunnar/Projects/blt/proc/rbr/blt1/blt2_rbr_ctd_cal_offsets.nc"
)
ctdcal_blt1 = xr.open_dataarray(
    "/Users/gunnar/Projects/blt/proc/rbr/blt1/blt1_rbr_ctd_cal_offsets.nc"
)

# %%
ctdcal.sel(sn=sn).data

# %% [markdown]
# Apply calibration offset to the test dataset.

# %%
tmpc = bp.thermistors.rbr_apply_ctd_offset(thermistor=tmp, sn=sn, ctdcal=ctdcal)

# %%
tmp.sel(time='2022-08-01 01:00').plot()
tmpc.sel(time='2022-08-01 01:00').plot()

# %% [markdown]
# Load mooring sensor info to know which sensor on which mooring.

# %%
path_to_thermistor_info = Path("/Users/gunnar/Projects/blt/moorings/thermistors/")
mavs3_rbr, mavs4_rbr = bp.thermistors.rbr_blt2_load_mooring_sensor_info(
    path_to_thermistor_info
)

no_mavs_rbr = True
if no_mavs_rbr:
    mavs3_rbr = mavs3_rbr.where(mavs3_rbr.Type != "MAVS Solo").dropna()
    mavs4_rbr = mavs4_rbr.where(mavs4_rbr.Type != "MAVS Solo").dropna()

# %% [markdown]
# *From BLT1 the option to cut time series short manually - probably don't need this here. Leaving it just in case as an empty dict.*

# %%
end_manually = dict()
# end_manually[72188] = np.datetime64('2021-08-05 00:00:00')

# %% [markdown]
# Test running the cut & cal function.

# %%
test = bp.thermistors.blt2_rbr_cut_and_cal(
    sn, level0_dir, level1_dir, path_to_thermistor_info, ctdcal, end_manually
)

# %%
ax = test.gv.plot()

# %% [markdown]
# Run for all RBRs and save to level 1 directory.

# %%
run_all = False

# %% [markdown]
# Currently missing Kurt's RBRs on the MAVS... Let's include a switch that excludes them and we can add them later on.

# %%
if run_all:
    sn_mavs3 = mavs3_rbr.index.to_numpy()
    for sni in sn_mavs3:
        logger.info(f"Saving {sni}")
        sni = int(sni)
        test = bp.thermistors.blt2_rbr_cut_and_cal(
            sni,
            level0_dir,
            level1_dir,
            path_to_thermistor_info,
            ctdcal,
            end_manually,
            no_mavs_rbr=True,
        )

    sn_mavs4 = mavs4_rbr.index.to_numpy()
    for sni in sn_mavs4:
        logger.info(f"Saving {sni}")
        sni = int(sni)
        test = bp.thermistors.blt2_rbr_cut_and_cal(
            sni,
            level0_dir,
            level1_dir,
            path_to_thermistor_info,
            ctdcal,
            end_manually,
            no_mavs_rbr=True,
        )

# %% [markdown]
# Success! Now we are just missing the RBRs that were integrated with the MAVS, I requested them from Kurt. These are:
# ```
# 201906, 201912, 201915, 202345, 202346, 202347, 202351, 202352
# ```
# Update: I have these now except for 202352 which I think did not produce data.

# %% [markdown]
# We have a few of Kurt's time series. Let's run these.
#
# ```
# 102974, 201846, 202304, 202305, 202306, 202307
# ```
#
# Note that we don't have a CTD cal for most of these. Let's use them from BLT1 where we have any.

# %%
mavs_on_hand = [102974, 201846, 202304, 202305, 202306, 202307]

# %% [markdown]
# Turns out we have a cal for 102974 but not for the remaining ones.

# %%
[sni in ctdcal_blt1.sn for sni in mavs_on_hand]

# %%
tmp = bp.thermistors.blt2_rbr_cut_and_cal(
    102974,
    level0_dir,
    level1_dir,
    path_to_thermistor_info,
    ctdcal_blt1,
    end_manually,
)

# %%
ax = tmp.gv.plot()

# %%
logger.warning(f"10297 only measures until {gv.time.datetime64_to_str(tmp.time[-1].data, unit='D')}")

# %%
tmp = bp.thermistors.blt2_rbr_cut_and_cal(
    201846,
    level0_dir,
    level1_dir,
    path_to_thermistor_info,
    ctdcal,
    end_manually,
)

# %%
ax = tmp.gv.plot()

# %%
for sni in mavs_on_hand[2:-1]:
    sni = int(sni)
    test = bp.thermistors.blt2_rbr_cut_and_cal(
        sni,
        level0_dir,
        level1_dir,
        path_to_thermistor_info,
        ctdcal,
        end_manually,
        no_mavs_rbr=False,
    )

# %% [markdown]
# 202307 appears to be broken starting in mid-December. Cut it short and save again.

# %%
tmp = bp.thermistors.blt2_rbr_cut_and_cal(
    202307,
    level0_dir,
    level1_dir,
    path_to_thermistor_info,
    ctdcal,
    end_manually,
    no_mavs_rbr=False,
)

# %%
file = next(level1_dir.glob("blt2*202307*.nc"))
tmp = xr.open_dataarray(file)
cut_time = np.datetime64("2021-12-18 19:00")

logger.warning(f"202307 only measures until {gv.time.datetime64_to_str(cut_time)}")
# save shortened time series
shortened = tmp.where(tmp.time < cut_time, drop=True).copy()
tmp.close()
shortened.to_netcdf(file)

# %% [markdown]
# Run 201906, 201912, 201915, 202345, 202346, 202347, 202351 as I now have these.

# %%
tmp = xr.open_dataarray("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc/201906_time_corrected.nc")

# %%
tmp2 = xr.open_dataarray("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS-BLT2/proc/201908.nc")

# %%
tmp.gv.tplot()

# %%
tmp2.gv.tplot()

# %%
tmp2.attrs["sampling period in s"]

# %%
td = tmp2.time.diff(dim="time")

# %%
td.sel(time=slice("2022-01-15 03:55:00", "2022-01-15 03:56:00")).plot(marker=".")

# %%
td.sel(time=slice("2022-01-15 03:55:00", "2022-01-15 03:56:00")).max().data.astype('<m8[ms]')

# %%
gaps = bp.thermistors.rbr_find_gaps(tmp)
print(len(gaps))
gaps.plot()

# %%
gaps.time

# %%
firstlonggap = bp.thermistors.find_first_long_gap(gaps)

# %%
np.isnat(firstlonggap)

# %%
longgap = bp.thermistors.rbr_find_first_long_gap(gaps)

# %%
# note here it's still 201908, further below we'll rename this to 201906
mavs_integrated = [201908, 201912, 201915, 202345, 202346, 202347, 202351]

# %%
sni = int(201906)
test = bp.thermistors.blt2_rbr_cut_and_cal(
    sni,
    level0_dir,
    level1_dir,
    path_to_thermistor_info,
    ctdcal,
    end_manually,
    no_mavs_rbr=False,
)

# %%
for sni in mavs_integrated[1:]:
    sni = int(sni)
    test = bp.thermistors.blt2_rbr_cut_and_cal(
        sni,
        level0_dir,
        level1_dir,
        path_to_thermistor_info,
        ctdcal,
        end_manually,
        no_mavs_rbr=False,
    )

# %% [markdown]
# Finally, let's rename 201906 to 201908 since that seems to be the right SN for this instrument.

# %%
tmp = xr.open_dataarray(level1_dir.joinpath("blt2_rbr_201908.nc"))
tmp.close()

# %%
tmp.attrs

# %%
tmp.attrs["SN"] = "201906"

# %%
tmp.to_netcdf(level1_dir.joinpath("blt2_rbr_201906.nc"))

# %%
# delete 201908
file_to_remove = level1_dir.joinpath("blt2_rbr_201908.nc")
if file_to_remove.exists():
    file_to_remove.unlink()
else:
    print(f"No such file: '{file_to_remove}'")

# %% [markdown]
# ## Plot all L1 time series for inspection

# %%
fig_dir = Path('blt2_inspect_level1')

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
run_all = False

# %%
if run_all:
    fig, ax = gv.plot.quickfig()
    for file in l1_files:
        tmp = xr.open_dataarray(file)
        plot_thermistor_time_series(tmp, ax)

# %% [markdown]
# Plot the MAVS-integrated time series (should also be included in `run_all` above, just keeping the code preserved in the following:

# %%
if False:
    fig, ax = gv.plot.quickfig()
    for sni in mavs_integrated:
        file = level1_dir.joinpath(f"blt2_rbr_{sni}.nc")
        tmp = xr.open_dataarray(file)
        plot_thermistor_time_series(tmp, ax)

# %%
