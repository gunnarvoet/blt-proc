# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] heading_collapsed=true
# #### Imports

# %% hidden=true
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import datetime
import gsw

import gvpy as gv
import gadcp

# load local functions
# import blt_adcp_proc_functions as nap
import bltproc as bp
# import blt

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

plt.ion()

# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# # BLT ADCP data processing

# %% [markdown]
# Starting May 10, 2022 I am keeping processing updates here.
#
# 2022-05-10: I noticed that I did not use the newly developed burst-averaging routine in `gadcp`. Updating this here.
#
# 2022-05-11: Reworking the depth gridding parameters. I had set up the depth grid such that the center of bins away from the ADCP were lining up pretty well with the depth vector used for gridding. While this is wanted for good interpolation, it means that whenever the ADCP is a bit deeper than that its data is not good for interpolating to the same bin anymore. I am now initially binning to 4m and then re-average to 16m for a smoother result.

# %% [markdown] heading_collapsed=true
# ## Set parameters

# %% [markdown] hidden=true
# Set the path to your local mooring data directory; this is the directory level that contains folders for individual moorings.

# %% hidden=true
blt_data = Path('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/')

# %% hidden=true
project = "BLT"

# %% [markdown] hidden=true
# Save the parameters to `parameters.yml` so we can read them with other functions and do not need to pass them as parameters every time.

# %% hidden=true
bp.adcp.save_params(path=blt_data, project=project)

# %% hidden=true
plotraw = True

# %% [markdown] heading_collapsed=true
# ## Read time offsets

# %% [markdown] hidden=true
# There are two files with time drift information, I must have done this at sea and then on shore again. The second file does not have the instrument type info which makes it easier to read as it is the same format as the time drift file that is used for the SBE37.

# %% hidden=true
time_offsets = bp.adcp.read_time_offsets()
time_offsets

# %% [markdown] heading_collapsed=true
# ## MP1

# %% hidden=true
sn = 24839
mooring = 'MP1'

# %% [markdown] hidden=true
# The depth grid limits are now determined automatically by `gadcp` based on the median depth of ADCP. Here we set a small bin size (a quarter of the ADCP bin size) to improve depth-averaging and will then bin-average back to 16m in the end.

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams, verbose=True)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 34 to 37.

# %% hidden=true
binmask = a.generate_binmask([34, 35, 36, 37])
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(a, savefig=True)

# %% [markdown] heading_collapsed=true
# ## MP2

# %% hidden=true
sn = 24839
mooring = 'MP2'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 34 to 37

# %% hidden=true
binmask = a.generate_binmask([34, 35, 36, 37])
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a)

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
data.u.where(data.pg>60).dropna(dim='depth', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MP3

# %% hidden=true
sn = 24839
mooring = 'MP3'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 34 to 37

# %% hidden=true
binmask = a.generate_binmask([34, 35, 36, 37])
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a)

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
data.u.where(data.pg>60).dropna(dim='depth', how='all').sel(time=slice('2022-07-01', '2022-07-14')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MAVS1

# %% [markdown] hidden=true
# Changes to previously processed dataset: Removing bins 0 and 1 from the processing due to contamination. Changed depth gridding due to bin-averaging pressure first.

# %% hidden=true
sn = 24608
mooring = 'MAVS1'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and 1 (sound bouncing off MAVS near the ADCP?) and 16 to 18 and update the processing instance with the editing parameters. We have 25m of wire between the buoy and the end of the first bin, the instrument may affect two bins here. We should stay away a bit further the next time around, and maybe try to have the first instrument at the center of a bin and not at the transition between two bins.

# %% hidden=true
binmask = a.generate_binmask([0, 1])
binmask[16:] = True
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a, savefig=True)

# %% [markdown] hidden=true
# We can filter data even further based on `pg` but leave this up to the data user.

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.u.where(a.ds.pg>60).dropna(dim='depth', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.pg.dropna(dim='depth', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MAVS2

# %% hidden=true
sn = 24606
mooring = 'MAVS2'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and maybe 1 (sound bouncing off MAVS near the ADCP?) and 16 to 18 and update the processing instance with the editing parameters. Actually, keep bin 1 for now, looks better than on MAVS1.

# %% hidden=true
binmask = a.generate_binmask([0])
binmask[16:] = True
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a, savefig=True)

# %% [markdown] heading_collapsed=true
# ## MAVS3

# %% hidden=true
sn = 24608
mooring = 'MAVS3'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and 1 (sound bouncing off MAVS near the ADCP?) and 14 to 18 and update the processing instance with the editing parameters. This is two bins earlier at the bottom than what we had on MAVS1 - I wonder if this is due to beam separation and being close to the canyon walls?

# %% hidden=true
binmask = a.generate_binmask([0, 1])
binmask[14:] = True
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a, savefig=True)

# %% [markdown] hidden=true
# We can filter data even further based on `pg` but leave this up to the data user.

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.u.where(a.ds.pg>10).dropna(dim='depth', how='all').sel(time=slice('2022-06-08', '2022-06-10')).gv.tplot(ax=ax)

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.pg.dropna(dim='depth', how='all').sel(time=slice('2022-06-08', '2022-06-10')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MAVS4

# %% hidden=true
sn = 24606
mooring = 'MAVS4'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and maybe 1 (sound bouncing off MAVS near the ADCP?) and 14 to 18 and update the processing instance with the editing parameters. Actually, keep bin 1 for now, looks better than on MAVS1.

# %% hidden=true
binmask = a.generate_binmask([0])
binmask[14:] = True
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.burst_average_ensembles()

# %% [markdown] hidden=true
# Interpolate / bin-average back to 16m resolution.

# %% hidden=true
a.rebin_dataset()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a, savefig=True)

# %% [markdown]
# ## TCHAIN

# %%
sn = 24607
mooring = 'TCHAIN'

# %%
dgridparams = dict(d_interval=4)

# %% [markdown]
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %%
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown]
# Plot beam statistics to find ADCP bins that need to be excluded.

# %%
a.plot_echo_stats()

# %%
binmask = a.generate_binmask([0])
binmask[8] = True
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %%
a.editparams

# %% [markdown]
# Burst-average.

# %%
a.burst_average_ensembles()

# %% [markdown]
# Interpolate / bin-average back to original 16m resolution.

# %%
a.rebin_dataset()

# %%
a.save_averaged_data()

# %%
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %%
if plotraw:
    bp.adcp.plot_raw_adcp(a, savefig=True)

# %% [markdown] heading_collapsed=true
# ## G1

# %% hidden=true
sn = 24839
mooring = 'G1'

# %% hidden=true
dgridparams = dict(d_interval=4)

# %% hidden=true
tgridparams = dict(dt_hours = 0.1)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = bp.adcp.ProcessBltADCP(mooring, sn, dgridparams=dgridparams, burst_average=False, tgridparams=tgridparams)

# %% hidden=true
a.tgridparams

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask last bin (64).

# %% hidden=true
binmask = a.generate_binmask([64])
editparams = dict(maskbins=binmask, min_correlation=20, pg_limit=30)
a.parse_editparams(editparams)

# %% hidden=true
a.orientation

# %% hidden=true
a.editparams

# %% hidden=true
a.average_ensembles()

# %% hidden=true
a.ds.u.sel(time='2022').gv.tplot(robust=True)

# %% hidden=true
a.ds.v.sel(time='2022').gv.tplot(robust=True)

# %% hidden=true
a.ds.w.sel(time='2022-08-09').gv.tplot(cmap='RdBu_r')

# %% hidden=true
a.ds.u.sel(time='2022-08-09').gv.tplot(robust=True)

# %% hidden=true
a.ds.v.sel(time='2022-08-09').gv.tplot(robust=True)

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = bp.adcp.plot_adcp(mooring, sn)
data.close()

# %% hidden=true
if plotraw:
    bp.adcp.plot_raw_adcp(a)
