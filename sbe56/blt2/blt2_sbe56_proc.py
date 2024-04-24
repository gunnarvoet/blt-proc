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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import tqdm

import gvpy as gv
import sbemoored as sbe

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'


# %%
import bltproc as bp

bp.thermistors.test()

# %%
pd.set_option("display.max_colwidth", None)

# %% [markdown]
# # BLT2 SBE56 Processing

# %% [markdown]
# This is for MAVS3/MAVS4 thermistors aka BLT2. We are using CTD calibration data from before the deployment (which were also used for calibrating BLT1 SBE56 thermistors).

# %%
B = bp.thermistors.blt2_sbe56_proc()

# %%
B.update_proc_info()

# %%
B.proc_info.head()

# %%
B.proc_next()

# %% [markdown]
# PROBLEMS:
# - 418: time offset appears to be wrong. The clock calibration time appear to be okay as it matches for sensors 916, 6417 that were in the same warm water dip batch. I am guessing that the wrong time was noted when downloading data. I adjusted the logger time in `blt2_sbe56_cal_info.csv` by adding 25 seconds to match the clock cal. The newly processed data then match the clock cal. I added a comment to the spreadsheet.

# %%
B.add_comment(
    418,
    "Adding 25s to Logger Time to match clock cal. Original logger time was 14:01:25.",
)
B.proc_info.loc[418].comment

# %%
B.proc_info.loc[418].comment

# %% [markdown]
# Now we are done with the basic processing.

# %% [markdown]
# Load all time series

# %%
B.load_all_nc()

# %% [markdown]
# Loop over all thermistors and find gaps.
# We want to know
# - if a thermistor has gaps at all
# - if and when the first long gap occurs
# - how many shorter gaps (<1h) occur before the first long gap

# %%
B.locate_gaps()

# %%
B.comment_short_gaps()

# %%
B.comment_long_gap_termination()

# %% [markdown]
# Make a note of the thermistors that do not have the warm water dip at the end because they were not measuring anymore. Most (actually for this deployment all?!) of them still had the clock running so should be fine.

# %%
B.comment_no_time_verification_at_end()

# %%
B.comment_terminates_early()

# %% [markdown]
# Remove nan's in Notes

# %%
B.comment_remove_nan_notes()

# %% [markdown]
# Show `proc_info` for all SNs that had any issues.

# %%
B.show_instrument_issues()

# %%
B.comments_to_latex_table()

# %% [markdown]
# ---

# %%
B.thermistor_info_path

# %%
B.load_mooring_sensor_info()

# %%
B.mavs3

# %% [markdown]
# ## cut and cal

# %%
B.load_ctd_cal()
B.load_mooring_sensor_info()

# %%
sbe37_mavs3 = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS3/SBE37/proc/SN12711/sbe37sm-rs232_03712711_2022_08_05.nc"
)

# %% [markdown]
# Find deployment and recovery times.

# %%
fig, ax = gv.plot.quickfig(grid=True)
sbe37_mavs3.p.sel(time=slice("2021-10-22 20:20", "2021-10-22 20:40")).plot(marker="o")

# %%
fig, ax = gv.plot.quickfig(grid=True)
sbe37_mavs3.p.sel(time=slice("2022-08-04 13:20", "2022-08-04 14:00")).plot(marker="o")

# %%
sbe37_mavs4 = xr.open_dataset(
    "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS4/SBE37/proc/SN12710/sbe37sm-rs232_03712710_2009_08_25.nc"
)

# %%
fig, ax = gv.plot.quickfig(grid=True)
sbe37_mavs4.p.sel(time=slice("2021-10-22 15:50", "2021-10-22 16:00")).plot(marker="o")

# %%
fig, ax = gv.plot.quickfig(grid=True)
sbe37_mavs4.p.sel(time=slice("2022-08-08 11:30", "2022-08-08 12:00")).plot(marker="o")

# %%
if False:
    B.generate_all_level1()

# %% [markdown]
# ## plot level 1 time series

# %% [markdown]
# ---

# %%
th = B.plot_level1(915)

# %% [markdown]
# Plot level 1 data and save all figures to

# %%
print(B.figure_out_level1)

# %%
if False:
    B.plot_all_level1()

# %% [markdown]
# What was the sampling frequency on all of these? 8s.

# %%
sn = np.array([ti.attrs["sn"] for ti in B.allnc])

sp = []
sp_range = []

for sni in tqdm.notebook.tqdm(sn):
    th = B.load_level1(sni)
    sp.append(th.gv.sampling_period)
    dt = th.time.diff(dim="time")
    sp_range.append(dt.max() - dt.min())

sp = np.array(sp)

fig, ax = gv.plot.quickfig()
ax.plot(sp)

# %% [markdown]
# No jumps in sampling frequency detected:

# %%
sp_range_ns = np.array([si.astype(float).data.item() for si in sp_range])

# %%
sp_range_ns
