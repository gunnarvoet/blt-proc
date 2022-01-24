# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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

import gvpy as gv
import sbemoored as sbe

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

# %config InlineBackend.figure_format = 'retina'


# %%
import bltproc as bp
bp.thermistors.test()

# %%
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # BLT SBE56 Processing

# %% [markdown]
# This is version 2 of the processing code where I have moved a number of processing functions into the `bltproc` module. The old version can still be found in the `old/` directory.

# %% [markdown]
# ## class

# %%
B = bp.thermistors.blt1_sbe56_proc()

# %%
B.update_proc_info()

# %%
B.proc_info.head()

# %%
B.proc_next()

# %% [markdown]
# Work out the 455 issue. We do not apply any time drift correction for this sensor during the initial processing and need to construct the time vector based on the clock calibration warm water dips.

# %%
B.fix_sn455()

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
# Make a note of the thermistors that do not have the warm water dip at the end because they were not measuring anymore. Most of them still had the clock running so should be fine.

# %%
B.comment_no_time_verification_at_end()

# %% [markdown]
# Add a comment if no data file exists, just to get rid of the ok mark.

# %%
B.comment_no_data()

# %% [markdown]
# Only one of the thermistors really ends too early (6432):

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
# ## cut and cal

# %%
B.load_ctd_cal()
B.load_mooring_sensor_info()

# %%
tmp = B.cut_and_cal(392)
sbe.sbe56.plot(tmp, figure_out=Path('.'))
plt.close()

# %%
B.generate_all_level1()

# %% [markdown]
# ## plot level 1 time series

# %%
th = B.plot_level1(915)

# %% [markdown]
# Plot level 1 data and save all figures to

# %%
print(B.figure_out_level1)

# %%
B.plot_all_level1()

# %% [markdown]
# ## plot spectra

# %% [markdown]
# This is still from v1 of this notebook and needs to be adapted to the new code structure if desired.

# %%
tmp = allnc[10]

# %%
tmp.size/8


# %%
def plot_spectrum(thermistor, window=8):
    g = thermistor.data
    sampling_period = thermistor.attrs['sampling period in s']
    if sampling_period == 0.5:
        sampling_period = 1
        omega_factor = 2
    else:
        omega_factor = 1
            
    Pcw, Pccw, Ptot, omega = gv.signal.psd(g, sampling_period, ffttype='t', window='hanning', tser_window=g.size/window)
    omega = omega * omega_factor
    f = 0.426

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5), constrained_layout=True)
    freqs = np.array([24 / (14 * 24), 24 / 12.4, 2 * 24 / 12.4, 4 * 24 / 12.4, f, 2 * f, 1, 24, 24/(24/24/6), 24/(24/24/60), 24/(24/24/60/6)])
    freq_labels = ["fortnightly", "M2", " \n2M2", "4M2", " \nf", " \n2f", "K1", "1h", "10m", "1m", "10s"]
    for freq in freqs:
        ax.vlines(freq, 1e-6, 1e4, color="C0", alpha=1, linestyle="-", linewidth=0.75)
    ax.plot(omega * (3600 * 24) / (2 * np.pi), np.real(Ptot), linewidth=1, color="0.2")
    ax.set(xscale="log", yscale="log", xlim=(2.1e-2, 5e4), ylim=(1e-7, 5e5), title=thermistor.attrs['SN'][-4:])
    ax = gv.plot.axstyle(ax, ticks="in", grid=True, spine_offset=10)
    gv.plot.gridstyle(ax, which="both")
    gv.plot.tickstyle(ax, which="both", direction="in")
    # ax2 = ax.twinx()
    ax2 = ax.secondary_xaxis(location="bottom")
    ax2 = gv.plot.axstyle(ax2, ticks="in", grid=False, spine_offset=30)
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_ticklabels([])
    ax2.minorticks_off()
    ax2.xaxis.set_ticks(freqs)
    ax2.xaxis.set_ticklabels(freq_labels);
    plot_name = figure_out.joinpath(f'spectrum_{thermistor.attrs["SN"]}')
    gv.plot.png(plot_name)


# %%
plot_spectrum(allnc[10])

# %%
# for thermistor in allnc:
#     plot_spectrum(thermistor)
