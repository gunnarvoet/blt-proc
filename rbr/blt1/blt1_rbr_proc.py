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
# %autosave 300
# %config InlineBackend.figure_format = 'retina'

# %%
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # BLT RBR Solo Processing
#
# - Convert .rsk files to netcdf files.
# - Correct for clock drift.
# - Plot clock calibration.

# %% [markdown]
# ## Set processing parameters

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")

rbr_dir = mooring_dir.joinpath("MAVS/RBRSolo")
data_raw = rbr_dir.joinpath("raw")
data_out = rbr_dir.joinpath("proc")
figure_out = rbr_dir.joinpath("fig")

# Create directories if needed
for d in [data_out, figure_out]:
    d.mkdir(exist_ok=True)


# %% [markdown]
# ## Get file name

# %%
def get_file_name(sn, data_raw, extension='rsk'):
    files = list(data_raw.glob(f'{sn:06}*.{extension}'))
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        raise OSError(f'more than one file for SN{sn} in {data_raw}')


# %% [markdown]
# ## Set up processing database

# %% [markdown]
# Clock calibration

# %%
def generate_proc_info():
    cal_file = mooring_dir.joinpath("blt1_rbrsolo_clock_calibration.txt")

    proc_info = pd.read_csv(
        cal_file,
        engine="python",
        header=0,
        delim_whitespace=True,
        parse_dates={
            "cal1": [1, 2],
            "cal2": [3, 4],
        },
        index_col="SN",
    )
#     proc_info.cal1 = pd.to_datetime(proc_info.cal2, errors='coerce')
    proc_info.cal2 = pd.to_datetime(proc_info.cal2, errors='coerce')

    n = proc_info.index.shape[0]

    proc_info = proc_info.assign(processed=np.tile(False, n))
    proc_info = proc_info.assign(raw_data_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(figure_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(comment=np.tile("ok", n))

    proc_info = proc_info.sort_index()
    return proc_info


# %% [markdown]
# Mark existing raw files

# %%
def update_proc_info():
    for g, v in proc_info.groupby('SN'):
        try:
            f = get_file_name(g, data_raw=data_raw, extension='rsk')
            if f is not None:
                proc_info.raw_data_exists.at[g] = True
            f = get_file_name(g, data_raw=data_out, extension='nc')
            if f is not None:
                proc_info.processed.at[g] = True
            f = get_file_name(g, data_raw=figure_out, extension='png')
            if f is not None:
                proc_info.figure_exists.at[g] = True
        except:
            pass


# %% [markdown]
# Extract calibration times

# %%
def get_cal_time(sn):
    cal1 = proc_info.loc[sn]['cal1'].to_datetime64()
    cal2 = proc_info.loc[sn]['cal2'].to_datetime64()
    return cal1, cal2


# %% [markdown]
# Process Files

# %%
def runproc(sn):
    cals = get_cal_time(sn)
    solofile = get_file_name(sn, data_raw)
    print(solofile.name)
    solo = rbr.solo.proc(
        solofile,
        data_out=data_out,
        apply_time_offset=True,
        figure_out=figure_out,
        cal_time=cals,
        show_plot=True,
    )


# %%
def proc_next():
    update_proc_info()
    sn = (
        proc_info.where(~proc_info.processed & proc_info.raw_data_exists)
        .dropna()
        .iloc[0]
        .name
    )
    runproc(sn)


# %%
proc_info = generate_proc_info()
update_proc_info()

# %%
proc_info.loc[72156]

# %% [markdown]
# ## process next

# %%
proc_next()


# %% [markdown]
# ## problems

# %%
def add_comment(sn, comment, proc_info):
    if type(sn) is float:
        sn = [sn]
    for si in sn:
        if proc_info.comment.at[si] == "ok":
            proc_info.comment.at[si] = comment
        if comment not in proc_info.comment.at[si]:
            addstr = "; "
            proc_info.comment.at[si] = proc_info.comment.at[si] + addstr + comment


# %%
sn = [72183]
comment = 'large time offset (seems ok though)'
add_comment(sn, comment, proc_info)

sn = [72188, 207286, 207287, 207288, 207289, 207300, 207301, 207302, 207303, 207304, 207305, 207306, 207307, 207308, 207309, 207375, 207376, 207377, 207378, 207379, 207380, 207381, 207382, 207383, ]
comment = 'time series terminating early'
add_comment(sn, comment, proc_info)

sn = [72188, 207286, 207287, 207288, 207289, 207300, 207301, 207302, 207303, 207304, 207305, 207306, 207307, 207308, 207309, 207375, 207376, 207377, 207378, 207379, 207380, 207381, 207382, 207383, ]
comment = 'no time offset recorded'
add_comment(sn, comment, proc_info)

# %%
print(proc_info.loc[72188].comment)

# %% [markdown]
# Show all instruments with problems

# %%
proc_info.where(proc_info.comment != 'ok').dropna()

# %%
proc_info.where(proc_info.comment != 'ok').dropna().to_csv('blt1_rbrsolo_proc_info_issues.csv')

# %%
proc_info.where(proc_info.comment != 'ok').dropna().index.to_numpy()

# %%
proc_info.to_csv('blt1_rbrsolo_proc_info.csv')

# %% [markdown]
# Save comments to a latex table.

# %%
proc_info.where(proc_info.comment != 'ok').dropna().to_latex(
    "/Users/gunnar/Projects/blt/proc/doc/rbrsolo_table.tex",
    columns=[
        "comment",
    ],
    header=["Processing Notes"],
    longtable=True,
    column_format="p{1.5cm}p{10cm}",
    multirow=False,
    caption="RBR Solo processing notes. The major issue was the early battery termination on all newly delivered thermistors. At the time of data processing the exact reason for the short lived batteries remains unclear.",
    label="tab:rbrsolo",
)
