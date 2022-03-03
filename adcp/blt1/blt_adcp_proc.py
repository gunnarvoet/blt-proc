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
import blt_adcp_proc_functions as nap
# import blt

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

plt.ion()

# %config InlineBackend.figure_format = 'retina'


# %% [markdown]
# # BLT ADCP data processing

# %% [markdown]
# ## Set parameters

# %% [markdown]
# Set the path to your local mooring data directory; this is the directory level that contains folders for individual moorings.

# %%
blt_data = Path('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/')

# %%
project = "BLT"

# %% [markdown]
# Save the parameters to `parameters.yml` so we can read them with other functions and do not need to pass them as parameters every time.

# %%
nap.save_params(path=blt_data, project=project)

# %%
plotraw = True

# %% [markdown] heading_collapsed=true
# ## Read time offsets

# %% [markdown] hidden=true
# There are two files with time drift information, I must have done this at sea and then on shore again. The second file does not have the instrument type info which makes it easier to read as it is the same format as the time drift file that is used for the SBE37.

# %% hidden=true
time_offsets = nap.read_time_offsets()

# %% [markdown] heading_collapsed=true
# ## MP1

# %% hidden=true
sn = 24839
mooring = 'MP1'

# %% hidden=true
# dgridparams = dict(dbot=2100, dtop=1300, d_interval=16)
# now generating top and bot automatically based on median depth of ADCP
dgridparams = dict(d_interval=16)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = nap.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 34 to 37

# %% hidden=true
binmask = a.generate_binmask([34, 35, 36, 37])
editparams = dict(maskbins=binmask)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.average_ensembles()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% hidden=true
a.meta_data

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(a, savefig=True)

# %% [markdown] heading_collapsed=true
# ## MP2

# %% hidden=true
sn = 24839
mooring = 'MP2'

# %% hidden=true
# dgridparams = dict(dbot=1700, dtop=1000, d_interval=16)
# now generating top and bot automatically based on median depth of ADCP
dgridparams = dict(d_interval=16)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = nap.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 34 to 37

# %% hidden=true
binmask = a.generate_binmask([34, 35, 36, 37])
editparams = dict(maskbins=binmask)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.average_ensembles()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% hidden=true
a.meta_data

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(mooring, sn)

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.u.where(a.ds.pg>60).dropna(dim='z', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MAVS1

# %% [markdown] hidden=true
# Changes to previously processed dataset: Removing bins 0 and 1 from the processing due to contamination. Changed depth gridding due to bin-averaging pressure first.

# %% hidden=true
sn = 24608
mooring = 'MAVS1'

# %% hidden=true
# dgridparams = dict(dbot=1650, dtop=1300, d_interval=16)
# now generating top and bot automatically based on median depth of ADCP
dgridparams = dict(d_interval=16)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = nap.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and 1 (sound bouncing off MAVS near the ADCP?) and 16 to 18 and update the processing instance with the editing parameters. We have 25m of wire between the buoy and the end of the first bin, the instrument may affect two bins here. We should stay away a bit further the next time around, and maybe try to have the first instrument at the center of a bin and not at the transition between two bins.

# %% hidden=true
print(a.meta_data.Bin1Dist)

# %% hidden=true
binmask = a.generate_binmask([0, 1])
binmask[16:] = True
editparams = dict(maskbins=binmask)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.average_ensembles()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% hidden=true
a.meta_data

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(a, savefig=True)

# %% [markdown] hidden=true
# We can filter data even further based on `pg` but leave this up to the data user.

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.u.where(a.ds.pg>60).dropna(dim='z', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)
gv.plot.png('test')

# %% hidden=true
fig, ax = gv.plot.quickfig(w=10)
a.ds.pg.where(a.ds.pg>60, drop=True).dropna(dim='z', how='all').sel(time=slice('2021-08-08', '2021-08-10')).gv.tplot(ax=ax)

# %% [markdown] heading_collapsed=true
# ## MAVS2

# %% hidden=true
sn = 24606
mooring = 'MAVS2'

# %% hidden=true
# dgridparams = dict(dbot=1480, dtop=1155, d_interval=16)
# now generating top and bot automatically based on median depth of ADCP
dgridparams = dict(d_interval=16)

# %% [markdown] hidden=true
# Generate a `ProcessBltADCP` instance that is based off the `gadcp.madcp.ProcessADCP` object.

# %% hidden=true
a = nap.ProcessBltADCP(mooring, sn, dgridparams=dgridparams)

# %% hidden=true
a.dgridparams

# %% hidden=true
a.meta_data

# %% hidden=true
a.files

# %% [markdown] hidden=true
# Plot beam statistics to find ADCP bins that need to be excluded.

# %% hidden=true
a.plot_echo_stats()

# %% [markdown] hidden=true
# Mask bins 0 and maybe 1 (sound bouncing off MAVS near the ADCP?) and 16 to 18 and update the processing instance with the editing parameters. Actually, keep bin 1 for now, looks better than on MAVS1.

# %% hidden=true
binmask = a.generate_binmask([0])
binmask[16:] = True
editparams = dict(maskbins=binmask)
a.parse_editparams(editparams)

# %% hidden=true
a.editparams

# %% [markdown] hidden=true
# Burst-average.

# %% hidden=true
a.average_ensembles(100, 300)

# %% hidden=true
ax = a.ds.u.gv.tplot()
ax = a.ds.pressure.gv.tplot(ax=ax)

# %% hidden=true
a.average_ensembles()

# %% hidden=true
a.save_averaged_data()

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(a, savefig=True)
