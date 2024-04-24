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
# # BLT RBR Solo MAVS Processing

# %% [markdown]
# Process temperature time series from RBR Solo that were integrated with the MAVS instruments.
#
# The clocks of these sensors have not been corrected for time drifts yet. I sent out an email to Kurt to ask him for the time drifts. I will read the processed time series, apply the time drift, and then save them again as I do not have the raw data on hand to rerun the whole processing.

# %% [markdown]
# ## Set processing parameters

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")

rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo")
data_out = rbr_dir.joinpath("proc")
figure_out = rbr_dir.joinpath("fig")

# %% [markdown]
# ## Read MAVS Solo

# %% [markdown]
# We need a mapping from MAVS SNs to RBR SNs.
# From Kurt's readme:
# ```
# 10290 and 10298 flooded.
# The temperature sensor on 10377 is non-functional after the first 7 days.  
# ```

# %%
mavsmap = [
    [10288, 90042],
    [10295, 201909],
    #     [10290, 202350],
    [10379, 201914],
    #     [10298, 202352],
    [10374, 201915],
    [10376, 201911],
    [10378, 201912],
    [10296, 202348],
    [10289, 202351],
    [10299, 202349],
#     [10377, 201910],
    [10375, 201913],
    [10373, 202345],
    [10372, 201906],
    [10297, 202346],
]


# %% [markdown]
# Low-pass filter and down-sample to 2Hz before saving? We can always go back to the higher frequency data if needed, but want to merge these with other 2Hz data anyways. Data are sampled at 5Hz... On the other hand, we are also interpolating to a common time vector and can simply do this from the 5Hz data.

# %%
def proc_mavs_rbr(mavssn, rbrsn):
    matfile = Path(f'/Volumes/piston2/blt/Moorings/BLT1/MAVS-ShipProc/{mavssn}_ship.mat')
    s = gv.io.loadmat(matfile)
    time = gv.time.mattime_to_datetime64(s.MAVSTime)
    t = xr.DataArray(s.temperature, coords=[time], dims=['time'])

    out_name = f'{rbrsn:06}_mavs_shipproc.nc'
    out_path = data_out.joinpath(out_name)
    print(out_path)
    t.to_netcdf(out_path)


# %%
if 0:
    for mavs in tqdm(mavsmap):
        proc_mavs_rbr(mavs[0], mavs[1])

# %% [markdown]
# ## Correct clock drift

# %%
mavsmap_all = [
    [10288, 90042],
    [10295, 201909],
    [10290, 202350],
    [10379, 201914],
    [10298, 202352],
    [10374, 201915],
    [10376, 201911],
    [10378, 201912],
    [10296, 202348],
    [10289, 202351],
    [10299, 202349],
    [10377, 201910],
    [10375, 201913],
    [10373, 202345],
    [10372, 201906],
    [10297, 202346],
]

mavsmap_dict = {m : r for m, r in mavsmap_all}

# %% [markdown]
# From Kurt's little write-up:
# ```
# All instruments were initially interrogated on 06 October 2021.
# ```

# %%
date = '2021-10-06'

# %%
m1_sn = [10288, 10295, 10290, 10379, 10298, 10374, 10376, 10378]
m1_gmt_time =  ['09:36:30', '09:45:35', '09:54:10', '09:49:30', '09:12:45', '09:17:45', '09:23:15', '09:30:45']
m1_inst_time = ['09:35:58', '09:46:36', 'N/A', '09:49:22', '09:13:05', '09:17:52', '09:23:19', '09:33:01']

# %%
m2_sn = [10296, 10289, 10299, 10377, 10375, 10373, 10372, 10297]
m2_gmt_time = ['08:26:30', '08:20:40', '07:25:00', '07:11:48', '08:39:30', '08:46:00', '08:51:00', '08:56:00']
m2_inst_time = ['08:26:38', '08:20:34', '07:27:17', '07:12:16', '08:39:45', '08:46:37', '08:51:51', '08:56:33']


# %%
def generate_timestamp(time):
    if time == 'N/A':
        return np.datetime64("NaT")
    else:
        datetimestr = f'{date} {time}'
        return np.datetime64(datetimestr)

def generate_timestamps(times):
    return [generate_timestamp(ti) for ti in times]


# %%
m1_gmt = generate_timestamps(m1_gmt_time)
m1_inst = generate_timestamps(m1_inst_time)

m2_gmt = generate_timestamps(m2_gmt_time)
m2_inst = generate_timestamps(m2_inst_time)

# %%
gmt = np.concatenate((m1_gmt, m2_gmt))
inst = np.concatenate((m1_inst, m2_inst))
mavs_sn = np.concatenate((m1_sn, m2_sn))
rbr_sn = [mavsmap_dict[m] for m in mavs_sn]

gmtd = {r : g for r, g in zip(rbr_sn, gmt)}
instd = {r : g for r, g in zip(rbr_sn, inst)}

# %% [markdown]
# Loop over these instruments:

# %%
# all RBRs
rbrsn = [m[1] for m in mavsmap]

# %%
rbr = rbrsn[0]

# %%
rbr

# %%
{'units': '°C',
 'long_name': 'temperature',
 'SN': 207286,
 'model': 'RBRsolo³',
 'firmware version': '1.054',
 'file': '207286_20211007_1555.rsk',
 'time drift in ms': -686937060429,
 'download time': 'N/A',
 'sample size': 17711599,
 'sampling period in s': 0.5,
 'time offset applied': 0}

# %%
tmp = xr.open_dataarray('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo/proc/201912_time_corrected.nc')


# %%
def correct_time(rbr):
    print('SN', rbr)
    file = data_out.joinpath(f'time_not_corrected_{rbr:06}_mavs_shipproc.nc')

    t = xr.open_dataarray(file)

    # We need to scale the time offset to the time when the time series actually stops.
    insttime = instd[rbr]
    utctime = gmtd[rbr]
    starttime = t.time[0].data
    endtime = t.time[-1].data
    overall_length = np.timedelta64(insttime - starttime, "s")
    length = np.timedelta64(endtime - starttime, "s")
    scale_factor = length / overall_length

    print(scale_factor)

    # calculate time offset
    offset = (
        np.timedelta64(insttime - utctime, "ms").astype("int")
        * scale_factor
    )

    print(offset/1e3, 's')

    t.attrs["time drift in ms"] = offset
    t.attrs["nvalues"] = t.size

    old_time = t.time.copy()
    time_offset_linspace = np.linspace(
        0, t.attrs["time drift in ms"], t.attrs["nvalues"]
    )
    # convert to numpy timedelta64
    # this format can't handle non-integers, so we switch to nanoseconds
    time_offset = [
        np.timedelta64(int(np.round(ti * 1e6)), "ns")
        for ti in time_offset_linspace
    ]
    new_time = old_time - time_offset

    t["time"] = new_time
    t.attrs["time offset applied"] = 1
    sample_size = len(tmp)
    t.attrs['sample size'] = sample_size
    sampling_period = tmp.time.diff(dim='time').median().data
    sampling_period = np.round(sampling_period.astype(int)/1e9, 2)
    t.attrs['sampling period in s'] = sampling_period
    t.attrs['units'] = '°C'
    t.attrs['long_name'] = 'temperature'
    t.attrs['SN'] = rbr,
    t.attrs['model'] = 'RBRsolo'
    t.attrs['firmware version'] = 'N/A'

    # save
    savename = data_out.joinpath(f'{rbr:06}_time_corrected.nc')
    t.to_netcdf(savename, mode='w')
    t.close()


# %%
for sn in tqdm(rbrsn[5:]):
    correct_time(sn)

# %%
