# -*- coding: utf-8 -*-
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
from tqdm.notebook import tqdm

# %% [markdown]
# # BLT2 RBR Solo MAVS Processing

# %% [markdown]
# Process temperature time series from RBR Solo that were integrated with the MAVS instruments.
#
# Not sure about how to handle clock drift.

# %% [markdown]
# Note: 201908 should be 201906 (there was no 201908 in my notes, the 201906 I show is paired up with the same MAVS as Kurts 201908 in his notes).

# %% [markdown]
# ## Set processing parameters

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")

rbr_dir = mooring_dir.joinpath("MAVS-BLT2")
mat_dir = rbr_dir.joinpath("mat")
data_out = rbr_dir.joinpath("proc")
figure_out = rbr_dir.joinpath("fig")

# %% [markdown]
# ## Read MAVS Solo

# %% [markdown]
# Low-pass filter and down-sample to 2Hz before saving? We can always go back to the higher frequency data if needed, but want to merge these with other 2Hz data anyways. Data are sampled at 5Hz... On the other hand, we are also interpolating to a common time vector and can simply do this from the 5Hz data.

# %%
files = sorted(mat_dir.glob("*.mat"))


# %%
def read_blt2_mavs_rbr(file):
    sn = int(file.stem)
    s = gv.io.loadmat(file)
    time = gv.time.mattime_to_datetime64(s.MAVSTime)
    t = xr.DataArray(s.temperature, coords=[time], dims=["time"])

    t.attrs = {
        "units": "°C",
        "long_name": "temperature",
        "SN": sn,
        "model": "RBRsolo",
        "firmware version": "N/A",
        "file": file.name,
        "time drift in ms": np.nan,
        "download time": "N/A",
        "sample size": len(t),
        "sampling period in s": t.gv.sampling_period,
        "time offset applied": 0,
    }

    return t


# %%
def proc_blt2_mavs_rbr(file):
    t = read_blt2_mavs_rbr(file)
    out_name = f"{t.SN:06}.nc"
    out_path = data_out.joinpath(out_name)
    print(out_path)
    t.to_netcdf(out_path)


# %%
if False:
    for file in tqdm(files):
        proc_blt2_mavs_rbr(file)

# %% [markdown]
# Note: Clock corrections happen in a different notebook.
