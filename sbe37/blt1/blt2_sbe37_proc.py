# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### Imports

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

import gvpy as gv
import sbemoored as sbe

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

plt.ion()

# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# # BLT2 SBE37 data processing

# %% [markdown]
# ## Mooring locations

# %%
loc = xr.open_dataset('/Users/gunnar/Projects/blt/moorings/blt_mooring_locations.nc')

# %% [markdown]
# ## Set paths

# %%
blt_data = Path('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/')


# %% [markdown]
# Generate data paths depending on mooring / instrument serial number.

# %%
def construct_37_paths(sn, mooring):
    data_raw = blt_data.joinpath(mooring).joinpath('SBE37/raw/SN{}'.format(sn))
    raw_file = list(data_raw.glob('*.cnv'))[0]
    data_out = blt_data.joinpath(mooring).joinpath('SBE37/proc/SN{}'.format(sn))
    if not data_out.exists():
        data_out.mkdir()
    fig_out = blt_data.joinpath(mooring).joinpath('SBE37/fig/')
    if not fig_out.exists():
        fig_out.mkdir()
    return data_raw, raw_file, data_out, fig_out


# %% [markdown]
# ## Read time offsets

# %%
offset_file = blt_data.joinpath('blt2_SBE37_time_offsets.txt')

# %%
time_offsets = pd.read_csv(offset_file, engine='python', header=0, delim_whitespace=True, parse_dates={'utc': [3, 4], 'inst': [1, 2]}, index_col='SN')

# %%
time_offsets

# %% [markdown]
# ## Meta data

# %%
meta = dict(
    project="BLT",
    funding_agency="NSF",
    processed_by="Gunnar Voet",
    contact="gvoet@ucsd.edu",
)

# %% [markdown]
# ## Process

# %% [markdown]
# ### 12709
#
# MAVS4

# %%
sn = 12709
mooring = 'MAVS4'
mloc = loc.sel(mooring=mooring)

# %%
meta['mooring'] = mooring

# %% [markdown]
# Process and save the whole time series without cutting or applying the time offset.

# %%
data_raw, raw_file, data_out, fig_out = construct_37_paths(sn, mooring)
insttime = time_offsets.loc[sn].inst.to_datetime64()
utctime = time_offsets.loc[sn].utc.to_datetime64()
lon = mloc.lon_actual.data
lat = mloc.lat_actual.data
# cuttime = gv.time.str_to_datetime64('2021-08-21 00:00')
cuttime = None

# %%
sn12709 = sbe.sbe37.proc(raw_file, data_out=data_out, figure_out=fig_out, lat=lat, lon=lon, meta=meta)

# %% [markdown]
# The microcats were reset after the CTD calibration casts and started sampling again on Oct 17. Read again starting at that date to avoid the gap. Nevermind, let's cut to the times when the instrument was at depth.

# %%
cut_beg = np.datetime64('2021-10-17 00:00:00')

# %%
sn12709.p.sel(time=slice('2021-10-22 15:00', '2021-10-22 17:00')).plot()

# %%
cut_beg = np.datetime64('2021-10-22 16:00:00')

# %%
sn12709.p.sel(time=slice('2022-08-08 11:00', '2022-08-08 12:00')).plot()

# %%
cut_end = np.datetime64('2022-08-08 11:30:00')

# %%
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %%
file_name[:file_name.find('.nc')]

# %% [markdown]
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %%
sn12709 = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cut_end, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %%
sbe.sbe37.clock_check(sn12709)

# %% [markdown]
# ### 12710
#
# MAVS4

# %%
sn = 12710
mooring = 'MAVS4'
mloc = loc.sel(mooring=mooring)

# %%
meta['mooring'] = mooring

# %% [markdown]
# Process and save the whole time series without cutting or applying the time offset.

# %%
data_raw, raw_file, data_out, fig_out = construct_37_paths(sn, mooring)
insttime = time_offsets.loc[sn].inst.to_datetime64()
utctime = time_offsets.loc[sn].utc.to_datetime64()
lon = mloc.lon_actual.data
lat = mloc.lat_actual.data

# %%
mcraw = sbe.sbe37.proc(raw_file, data_out=data_out, figure_out=fig_out, lat=lat, lon=lon, meta=meta)

# %% [markdown]
# Set time for cut netcdf file.

# %%
cut_beg = np.datetime64('2021-10-22 16:00:00')

# %%
cut_end = np.datetime64('2022-08-08 11:30:00')

# %%
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %%
file_name[:file_name.find('.nc')]

# %% [markdown]
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %%
mc = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cut_end, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %%
sbe.sbe37.clock_check(mc)

# %% [markdown]
# ### 12711
#
# MAVS3

# %%
sn = 12711
mooring = 'MAVS3'
mloc = loc.sel(mooring=mooring)

# %%
meta['mooring'] = mooring

# %% [markdown]
# Process and save the whole time series without cutting or applying the time offset.

# %%
data_raw, raw_file, data_out, fig_out = construct_37_paths(sn, mooring)
insttime = time_offsets.loc[sn].inst.to_datetime64()
utctime = time_offsets.loc[sn].utc.to_datetime64()
lon = mloc.lon_actual.data
lat = mloc.lat_actual.data
# cuttime = gv.time.str_to_datetime64('2021-08-21 00:00')
cuttime = None

# %%
mcraw = sbe.sbe37.proc(raw_file, data_out=data_out, figure_out=fig_out, cut_end=cuttime, lat=lat, lon=lon)

# %% [markdown]
# Set time for cut netcdf file.

# %%
mcraw.p.sel(time=slice('2021-10-22 20:30', '2021-10-22 22:00')).plot()

# %%
cut_beg = np.datetime64('2021-10-22 21:00:00')

# %%
mcraw.p.sel(time=slice('2022-08-04 13:00', '2022-08-04 16:00')).plot()

# %%
cut_end = np.datetime64('2022-08-04 13:30:00')

# %% [markdown]
# Set name for cut netcdf file.

# %%
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %%
file_name[:file_name.find('.nc')]

# %% [markdown]
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %%
mc = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cut_end, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %%
sbe.sbe37.clock_check(mc)
