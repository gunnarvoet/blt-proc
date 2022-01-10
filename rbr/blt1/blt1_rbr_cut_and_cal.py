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

# %% hidden=true
import bltproc as bp
bp.thermistors.test()

# %% hidden=true
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # BLT RBR Solo Cut & Cal

# %% [markdown]
# - Cut RBR themistor time series to time at depth and before data start to drop out for some sensors.
# - Apply offsets determined in CTD calibration casts.

# %% [markdown]
# Most of the code moved the `bltproc.thermistors`.

# %% [markdown] heading_collapsed=true
# ### Set processing parameters

# %% hidden=true
# We are skipping a number of cells. Run them by setting the following to True.
run_all = False

# %% hidden=true
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo")
level0_dir = rbr_dir.joinpath("proc")
# directory for level 1 processed data
level1_dir = rbr_dir.joinpath("proc_L1")

# %% [markdown]
# ## Cut and save calibrated time series

# %% [markdown]
# We want to save a level 1 dataset just as we did for the SBE56. This means determining the end of the good part of the time series, either where it stops or where it starts showing gaps due to the battery dropping out, and to apply the CTD calibration.

# %%
# read a test unit
sn = 207286
tmp = bp.thermistors.rbr_load_proc_level0(sn, level0_dir)

# %% [markdown]
# Find end times

# %%
last_time = bp.thermistors.rbr_find_last_time_stamp(tmp)
print(last_time)

# %%
tdi = bp.thermistors.rbr_find_gaps(tmp)
if len(tdi) > 0:
    print(tdi.time.data-tdi.data)
    print([np.timedelta64(t, 'h') for t in tdi.data])

# %% [markdown]
# Like with the SBE56, we stop the time series where gaps become longer than one hour.

# %%
t = bp.thermistors.rbr_find_first_long_gap(tdi)

# %%
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
# Test running the cut & cal function.

# %%
test = bp.thermistors.rbr_cut_and_cal(
    sn, level0_dir, level1_dir, path_to_thermistor_info, ctdcal
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
