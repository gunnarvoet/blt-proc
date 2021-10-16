# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
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
# # BLT1 SBE37 data processing

# %% [markdown]
# ## Mooring locations

# %%
loc = xr.open_dataset('/Users/gunnar/Projects/blt/moorings/blt1_mooring_locations.nc')

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
offset_file = blt_data.joinpath('blt1_SBE37_time_offsets.txt')

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

# %% [markdown] heading_collapsed=true
# ### 12709
#
# MAVS1

# %% hidden=true
mloc = loc.isel(mooring=1)
sn = 12709
mooring = 'MAVS1'

# %% hidden=true
meta['mooring'] = mooring

# %% [markdown] hidden=true
# Process and save the whole time series without cutting or applying the time offset.

# %% hidden=true
data_raw, raw_file, data_out, fig_out = construct_37_paths(sn, mooring)
insttime = time_offsets.loc[sn].inst.to_datetime64()
utctime = time_offsets.loc[sn].utc.to_datetime64()
lon = mloc.lon_actual.data
lat = mloc.lat_actual.data
# cuttime = gv.time.str_to_datetime64('2021-08-21 00:00')
cuttime = None

# %% hidden=true
sn12709 = sbe.sbe37.proc(raw_file, data_out=data_out, figure_out=fig_out, lat=lat, lon=lon, meta=meta)

# %% [markdown] hidden=true
# Shortly before data drop out the clock becomes unstable:

# %% hidden=true
cutoff = sbe.sbe37.clock_check(sn12709)

# %% hidden=true
fig, ax = gv.plot.quickfig()
sn12709.sel(time=slice('2021-08-20 02:00', '2021-08-22 07:00')).p.plot(linewidth=0.75);
sn12709.sel(time=slice('2021-08-20 02:00', cutoff)).p.plot(linewidth=0.75);

# %% [markdown] hidden=true
# Read again with cutoff time, apply clock offset, cut beginning of time series, and save as final.

# %% hidden=true
fig, ax = gv.plot.quickfig()
sn12709.sel(time=slice('2021-07-06 14:15', '2021-07-06 14:35')).p.plot(linewidth=0.75, marker='o');

# %% hidden=true
fig, ax = gv.plot.quickfig()
sn12709.sel(time=slice('2021-07-06 14:28', '2021-07-06 14:31')).p.plot(linewidth=0.75, marker='o');

# %% [markdown] hidden=true
# Set time for cut netcdf file.

# %% hidden=true
cut_beg = np.datetime64('2021-07-06 14:30:00')

# %% [markdown] hidden=true
# Set name for cut netcdf file.

# %% hidden=true
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %% hidden=true
file_name[:file_name.find('.nc')]

# %% [markdown] hidden=true
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %% hidden=true
sn12709 = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cutoff, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %% [markdown] hidden=true
# Now the clock  check should fail:

# %% hidden=true
sbe.sbe37.clock_check(sn12709)

# %% [markdown]
# ### 12710
#
# MAVS2

# %%
mloc = loc.isel(mooring=3)
sn = 12710
mooring = 'MAVS2'

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
# Process and save the whole time series without cutting or applying the time offset.

# %% [markdown]
# Shortly before data drop out the clock becomes unstable:

# %%
cutoff = sbe.sbe37.clock_check(mcraw)

# %%
t0 = cutoff - np.timedelta64(1, 'D')
t1 = cutoff + np.timedelta64(1, 'D')
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice(t0, t1)).p.plot(linewidth=0.75);
mcraw.sel(time=slice(t0, cutoff)).p.plot(linewidth=0.75);

# %%
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice('2021-07-07 13:31', '2021-07-07 13:42')).p.plot(linewidth=0.75, marker='o');

# %%
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice('2021-07-07 13:41', '2021-07-07 14:00')).p.plot(linewidth=0.75, marker='o');

# %% [markdown]
# Set time for cut netcdf file.

# %%
cut_beg = np.datetime64('2021-07-07 13:50:00')

# %% [markdown]
# Set name for cut netcdf file.

# %%
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %%
file_name[:file_name.find('.nc')]

# %% [markdown]
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %%
mc = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cutoff, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %% [markdown]
# Now the clock  check should fail:

# %%
sbe.sbe37.clock_check(mc)

# %% [markdown]
# ### 12711
#
# MAVS2

# %%
mloc = loc.isel(mooring=3)
sn = 12711
mooring = 'MAVS2'

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
# Shortly before data drop out the clock becomes unstable:

# %%
cutoff = sbe.sbe37.clock_check(mcraw, timedelta=200)

# %%
t0 = cutoff - np.timedelta64(12, 'h')
t1 = cutoff + np.timedelta64(12, 'h')
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice(t0, t1)).p.plot(linewidth=0.75);
mcraw.sel(time=slice(t0, cutoff)).p.plot(linewidth=0.75);

# %%
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice('2021-07-07 13:31', '2021-07-07 13:42')).p.plot(linewidth=0.75, marker='o');

# %%
fig, ax = gv.plot.quickfig()
mcraw.sel(time=slice('2021-07-07 13:41', '2021-07-07 14:00')).p.plot(linewidth=0.75, marker='o');

# %% [markdown]
# Set time for cut netcdf file.

# %%
cut_beg = np.datetime64('2021-07-07 13:50:00')

# %% [markdown]
# Set name for cut netcdf file.

# %%
file_name = f'BLT1_{mooring}_SBE37_SN{sn}.nc'

# %%
file_name[:file_name.find('.nc')]

# %% [markdown]
# Re-read the raw data, this time cut at end and beginning and apply the time offset.

# %%
meta['mooring'] = mooring

# %%
mc = sbe.sbe37.proc(raw_file, insttime, utctime, data_out=data_out, file_name=file_name, figure_out=fig_out, cut_end=cutoff, cut_beg=cut_beg, lat=lat, lon=lon, meta=meta)

# %% [markdown]
# Now the clock  check should fail:

# %%
sbe.sbe37.clock_check(mc)
