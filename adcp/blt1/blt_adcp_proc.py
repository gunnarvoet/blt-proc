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

# %% [markdown]
# #### Imports

# %%
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

# %% [markdown]
# ## Read time offsets

# %% [markdown]
# There are two files with time drift information, I must have done this at sea and then on shore again. The second file does not have the instrument type info which makes it easier to read as it is the same format as the time drift file that is used for the SBE37.

# %%
time_offsets = nap.read_time_offsets()

# %%
time_offsets

# %%
insttime = time_offsets.loc[sn].inst

# %%
time_offsets.query('Mooring=="MP1"').loc[24839]

# %% [markdown] heading_collapsed=true
# ## MP1

# %% hidden=true
sn = 24839
mooring = 'MP1'

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(mooring, sn)

# %% hidden=true
r = nap.plot_echo_stats(mooring, sn)

# %% [markdown] hidden=true
# mask bins 34 to 37

# %% hidden=true
binmask = (r.bin > 33).data

# %% hidden=true
editparams = dict(maskbins=binmask)

# %% hidden=true
dgridparams = dict(dbot=2100, dtop=1300, d_interval=16)

# %% hidden=true
data = nap.process_adcp(mooring, sn, dgridparams, editparams=editparams, n_ensembles=None, save_nc=True)

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% [markdown] heading_collapsed=true
# ## MP2

# %% hidden=true
sn = 24839
mooring = 'MP2'

# %% hidden=true
if plotraw:
    nap.plot_raw_adcp(mooring, sn)

# %% hidden=true
r = nap.plot_echo_stats(mooring, sn)

# %% [markdown] hidden=true
# mask bins 34 to 37

# %% hidden=true
binmask = (r.bin > 33).data

# %% hidden=true
editparams = dict(maskbins=binmask)

# %% hidden=true
dgridparams = dict(dbot=1700, dtop=1000, d_interval=16)

# %% hidden=true
data = nap.process_adcp(mooring, sn, dgridparams, editparams=editparams, n_ensembles=None, save_nc=True)

# %% hidden=true
data = nap.plot_adcp(mooring, sn)

# %% [markdown]
# ## MAVS1

# %%
sn = 24608
mooring = 'MAVS1'

# %%
if plotraw:
    nap.plot_raw_adcp(mooring, sn)

# %%
r = nap.plot_echo_stats(mooring, sn)

# %% [markdown]
# mask bins 16 to 18

# %%
binmask = (r.bin > 15).data

# %%
# binmask[0] = True

# %%
editparams = dict(maskbins=binmask)

# %%
dgridparams = dict(dbot=1650, dtop=1300, d_interval=16)

# %%
data = nap.process_adcp(mooring, sn, dgridparams, editparams=editparams, n_ensembles=None, save_nc=True)

# %%
data = nap.plot_adcp(mooring, sn)

# %%
data

# %%
fig, ax = gv.plot.quickfig()
data.sel(time="2021-08-01").u.where(data.z < 1400, drop=True).plot()
(-1 * gsw.z_from_p(data.sel(time="2021-08-01").pressure, data.lat)).plot()
ax.invert_yaxis()

# %%
data.u.mean(dim='time').plot(marker='o')

# %%
d = data.sel(time='2021-08').w

# %%
fig, ax = gv.plot.quickfig(w=3, h=5)
h = ax.violinplot(
    d.data.transpose(),
    positions=d.z.data,
    vert=False,
    widths=15,
    showmeans=True,
    showmedians=False,
    points=100,
)
ax.invert_yaxis()

# %% [markdown]
# ## MAVS2

# %%
sn = 24606
mooring = 'MAVS2'

# %%
if plotraw:
    nap.plot_raw_adcp(mooring, sn)

# %%
r = nap.plot_echo_stats(mooring, sn)

# %% [markdown]
# mask bin 0 and bins 16 to 18

# %%
binmask = (r.bin > 15).data

# %%
# binmask[0] = True

# %%
editparams = dict(maskbins=binmask)

# %%
dgridparams = dict(dbot=1480, dtop=1155, d_interval=16)

# %%
data = nap.process_adcp(mooring, sn, dgridparams, editparams=editparams, n_ensembles=None, save_nc=True)

# %%
data.close()

# %%
data = nap.plot_adcp(mooring, sn)

# %%
fig, ax = gv.plot.quickfig()
data.sel(time="2021-08-01").u.where(data.z < 1230, drop=True).plot()
(-1 * gsw.z_from_p(data.sel(time="2021-08-01").pressure, data.lat)).plot()
ax.invert_yaxis()

# %%
fig, ax = gv.plot.quickfig()
data.sel(time="2021-08-01").u.where(data.z < 1230, drop=True).plot()
(-1 * gsw.z_from_p(data.sel(time="2021-08-01").pressure, data.lat)).plot()
ax.invert_yaxis()

# %%
data.u.mean(dim='time').plot(marker='o')
