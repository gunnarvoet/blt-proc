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
import scipy
import xarray as xr
import pandas as pd
from pathlib import Path
import pyrsktools
import os

import gvpy as gv

# %reload_ext autoreload
# %autoreload 2

# %config InlineBackend.figure_format = 'retina'

# %%
gv.plot.helvetica()

# %%
from tqdm.notebook import tqdm

# %% [markdown]
# # BLT2 RBR Solo MAVS Check Clocks

# %% [markdown]
# Correlate last day of time series with corresponding segment from neighboring, clock-calibrated thermistors.

# %% [markdown]
# Strategy: Use the lag finder (v0 works, the other ones not quite yet) to determine lags for a number of time windows and both neighbors. Then calculate the average and use this to adjust the time vector.

# %% [markdown]
# Actually, let's list all RBRs from the MAVS and their neighbors (not done yet...).

# %%
neighbors = {
    "201908": [207378, 72212],
    "201912": [395, 6447],
    "201915": [421, 6433],
    "202345": [207306, 72154],
    "202346": [72209, 72213],
    "202347": [6415, 102969],
    "202351": [6431, 6428],
}


# %%
def load_mavs_rbr(sn):
    mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
    rbr_dir = mooring_dir.joinpath("MAVS-BLT2")
    proc_dir = rbr_dir.joinpath("proc")
    da = xr.open_dataarray(proc_dir_mavs.joinpath(f"{sn}.nc"))
    da.attrs["SN"] = str(sn)
    return da


# %%
def load_neighbor(sn):
    type = "rbr" if sn > 9999 else "sbe"
    match type:
        case "rbr":
            proc_dir = Path(
                "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc"
            )
            file = next(proc_dir.glob(f"{sn:06d}*.nc"))
        case "sbe":
            proc_dir = Path(
                "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/SBE56_2/proc"
            )
            file = next(proc_dir.glob(f"SBE056{sn:05d}*.nc"))
    da = xr.open_dataarray(file)
    return da


# %%
def load_triplet(mavs_rbr_sn, na, nb):
    a = load_neighbor(na)
    b = load_neighbor(nb)
    t = load_mavs_rbr(mavs_rbr_sn)
    return a, b, t


# %%
def lag_finder_detrend(y1, y2, sr):
    # Detrend the time series data
    y1 = detrend_dataarray(y1)
    y2 = detrend_dataarray(y2)

    # Calculate the cross-correlation
    corr = scipy.signal.correlate(y2, y1, mode="same") / np.sqrt(
        scipy.signal.correlate(y1, y1, mode="same")[int(len(y1) / 2)]
        * scipy.signal.correlate(y2, y2, mode="same")[int(len(y2) / 2)]
    )

    # Find the lag at which the cross-correlation is maximized
    delay_arr = np.linspace(-0.5 * len(y1) / sr, 0.5 * len(y1) / sr, len(y1))
    delay = delay_arr[np.argmax(corr)]
    # print("y2 is " + str(delay) + " behind y1")
    return delay


# Example usage:
# y1 and y2 are your time series data
# sr is your sampling rate
# lag_finder(y1, y2, sr)

# %%
def detrend_dataarray(da):
    detrended_da = xr.apply_ufunc(
        scipy.signal.detrend,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )
    return detrended_da


# %%
def find_lags(y1, y2, dt):
    delays = []
    t0 = y1.time[0].data
    t1 = y1.time[-1].data
    offset_counter = np.timedelta64(0, "h")
    t_end = t0 + dt
    while t_end < t1:
        ts = slice(t0 + offset_counter, t0 + offset_counter + dt)
        delay = lag_finder_detrend(
            y1.sel(time=ts, drop=True), y2.sel(time=ts, drop=True), 0.5
        )
        delays.append(delay)
        offset_counter += dt
        t_end += dt
    return np.array(delays)


# %%
def find_time_offset(rbr_mavs, neighbor0, neighbor1, window_size_hours=72, maxdev_s=20):
    tt = [neighbor0, neighbor1, rbr_mavs]

    t_start = t.time[0].data + np.timedelta64(1, "D")
    t_end = t.time[-1].data
    new_time = np.arange(t_start, t_end, dtype="datetime64[2s]").astype("<M8[ns]")

    tts = [
        ti.sel(
            time=slice(
                new_time[0] - np.timedelta64(1, "m"),
                new_time[-1] + np.timedelta64(1, "m"),
            ),
            drop=True,
        ).interp(time=new_time)
        for ti in tt
    ]

    dt = np.timedelta64(window_size_hours, "h")
    delaysa = find_lags(tts[0], tts[2], dt)
    delaysb = find_lags(tts[1], tts[2], dt)
    delays = (delaysa + delaysb) / 2

    # fit the combined offsets from both neighboring thermistors
    fit = np.polynomial.Polynomial.fit(np.arange(len(delays)), delays, deg=1)
    # generate values as with polyval
    xvalues, yvalues = fit.linspace(n=len(delays))
    offset = yvalues[-1]

    ya, yb = fit.convert().coef
    # print(f"initial offset: {ya:1.1f}s")
    # print(f"offset at end: {offset:1.1f}s")

    # exclude individual offsets that are more than maxdev_s away from the linear fit
    deva = np.absolute(yvalues - delaysa)
    devb = np.absolute(yvalues - delaysb)

    delaysaa = np.where(deva < maxdev_s, delaysa, np.nan)
    delaysbb = np.where(devb < maxdev_s, delaysb, np.nan)
    delays2 = np.nanmean([delaysaa, delaysbb], axis=0)
    # delays2 = (delaysaa + delaysbb) / 2
    mask = ~np.isnan(delays2)

    fit2 = np.polynomial.Polynomial.fit(
        np.arange(len(delays2))[mask], delays2[mask], deg=1
    )
    xvalues2, yvalues2 = fit2.linspace(n=len(delays2))
    offset2 = yvalues2[-1]

    ya2, yb2 = fit2.convert().coef
    print(f"initial offset: {ya2:1.1f}s")
    print(f"offset at end: {offset2:1.1f}s")

    fig, ax = gv.plot.quickfig(fs=12, grid=True)
    ax.plot(delaysa, color="0.5")
    ax.plot(delaysb, color="0.5")
    ax.plot(delays, color="C0", linewidth=2, linestyle="", marker="o")
    ax.plot(delays2, color="m", linewidth=2, linestyle="", marker="o")
    ax.plot(xvalues, yvalues, color="C0", linewidth=2, linestyle="--")
    ax.plot(xvalues2, yvalues2, color="m", linewidth=2, linestyle="--")
    ax.set(xlabel="window", ylabel="time offset [s]")
    if "SN" in rbr_mavs.attrs:
        ax.set(title=f"SN {rbr_mavs.attrs['SN']}")

    return offset2, ya2


# %%
def correct_time(rbr, offset_in_s):
    t = rbr.copy()
    print("SN", t.SN)

    print(offset_in_s, "s")

    t.attrs["time drift in ms"] = offset_in_s * 1e3

    old_time = t.time.copy()
    time_offset_linspace = np.linspace(
        0, t.attrs["time drift in ms"], t.attrs["sample size"]
    )
    # convert to numpy timedelta64
    # this format can't handle non-integers, so we switch to nanoseconds
    time_offset = [
        np.timedelta64(int(np.round(ti * 1e6)), "ns") for ti in time_offset_linspace
    ]
    new_time = old_time - time_offset

    t["time"] = new_time
    t.attrs["time offset applied"] = 1
    return t


# %%
def plot_corrected(t, t_c, nb, end=True):
    if end:
        t_end = t.time[-1].data
        delta = np.timedelta64(30, "m")
        sl = slice(t_end - delta, t_end)
    else:
        t_start = t.time[0].data + np.timedelta64(20, "D")
        delta = np.timedelta64(30, "m")
        sl = slice(t_start, t_start + delta)

    
    ax = detrend_dataarray(nb.sel(time=sl)).gv.plot(
        linewidth=1, color="0.5", label="neighbor sensor"
    )
    ax = detrend_dataarray(t.sel(time=sl)).gv.plot(
        linewidth=1, ax=ax, color="C0", alpha=0.5, label="not corrected"
    )
    ax = detrend_dataarray(t_c.sel(time=sl)).gv.plot(
        linewidth=1, ax=ax, color="C0", label="clock drift applied"
    )
    ax.set(title=f"clock correction {tmp.SN}")
    ax.legend()


# %%
def save_corrected(t):
    proc_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc")
    savename = proc_dir.joinpath(f"{t.SN:06}_time_corrected.nc")
    t.to_netcdf(savename)


# %% [markdown]
# Run for all integrated sensors

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
# MAVS RBRs
rbr_dir = mooring_dir.joinpath("MAVS-BLT2")
proc_dir_mavs = rbr_dir.joinpath("proc")
# other processed RBRs
proc_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc")

# %% [markdown]
# ## 201908

# %%
mavs_sn = 201908
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 201912

# %%
mavs_sn = 201912
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 201915

# %%
mavs_sn = 201915
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 202345

# %%
mavs_sn = 202345
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 202346

# %%
mavs_sn = 202346
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 202347

# %%
mavs_sn = 202347
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %%
t_c = correct_time(t, offset)

# %%
plot_corrected(t, t_c, a)
gv.plot.png(f"clock_correction_verify_{t_c.SN}")

# %%
plot_corrected(t, t_c, a, end=False)

# %%
save_corrected(t_c)

# %% [markdown]
# ## 202351

# %%
mavs_sn = 202351
na, nb = neighbors[str(mavs_sn)]
a, b, t = load_triplet(mavs_sn, na, nb)

# %%
offset, initial_offset = find_time_offset(t, a, b)
gv.plot.png(f"determine_clock_offset{t.SN}")

# %% [markdown]
# No need to adjust the time vector for this one.

# %%
# t_c = correct_time(t, offset)

# %%
save_corrected(t)

# %% [markdown]
# ---
# ## dev

# %% [markdown]
# Let's start out with one of them.
# 201908 is 201906 in my notation. It's on MAVS3. Neighboring Solos are 207378 and 102984.
# Load them.

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")
# MAVS RBRs
rbr_dir = mooring_dir.joinpath("MAVS-BLT2")
proc_dir_mavs = rbr_dir.joinpath("proc")
# other processed RBRs
proc_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo_2/proc")
# load time series
a = xr.open_dataarray(proc_dir.joinpath("207378_20220806_1352.nc"))
b = xr.open_dataarray(proc_dir.joinpath("102984_20220806_1752.nc"))
t = xr.open_dataarray(proc_dir_mavs.joinpath("201908.nc"))

# %%
tt = [a, b, t]

# %% [markdown]
# Interpolate to a 2s time vector. I don't think we have to lowpass-filter the MAVS data for this.

# %%
t_start = t.time[0].data + np.timedelta64(1, "D")
t_end = t.time[-1].data
new_time = np.arange(t_start, t_end, dtype="datetime64[2s]").astype("<M8[ns]")

# %%
tts = [
    ti.sel(
        time=slice(
            new_time[0] - np.timedelta64(1, "m"), new_time[-1] + np.timedelta64(1, "m")
        ),
        drop=True,
    ).interp(time=new_time)
    for ti in tt
]

# %% [markdown]
# What does a positive offset mean? It means the MAVS clock is ahead of the neighbors clock. Here we subtract the offset time from the MAVS time series and everything seems to line up.

# %%
delta = np.timedelta64(2, "D")
sl = slice(t_end - delta - np.timedelta64(20, "m"), t_end - delta)

# %%
# %matplotlib ipympl

# %%
tmp.SN

# %%
ax = detrend_dataarray(tt[0].sel(time=sl)).gv.plot(
    linewidth=1, color="0.5", label="neighor sensor"
)
ax = detrend_dataarray(tt[2].sel(time=sl)).gv.plot(
    linewidth=1, ax=ax, color="C0", alpha=0.5, label="not corrected"
)
tmp = tt[2].copy()
tmp["time"] = tmp.time - np.timedelta64(83, "s")
ax = detrend_dataarray(tmp.sel(time=sl)).gv.plot(
    linewidth=1, ax=ax, color="C0", label="clock drift applied"
)
ax.set(title=f"clock correction {tmp.SN}")
ax.legend()
gv.plot.png(f"clock_drift_correction_example_{tmp.SN}")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

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

mavsmap_dict = {m: r for m, r in mavsmap_all}

# %% [markdown]
# From Kurt's little write-up:
# ```
# All instruments were initially interrogated on 06 October 2021.
# ```

# %%
date = "2021-10-06"

# %%
m1_sn = [10288, 10295, 10290, 10379, 10298, 10374, 10376, 10378]
m1_gmt_time = [
    "09:36:30",
    "09:45:35",
    "09:54:10",
    "09:49:30",
    "09:12:45",
    "09:17:45",
    "09:23:15",
    "09:30:45",
]
m1_inst_time = [
    "09:35:58",
    "09:46:36",
    "N/A",
    "09:49:22",
    "09:13:05",
    "09:17:52",
    "09:23:19",
    "09:33:01",
]

# %%
m2_sn = [10296, 10289, 10299, 10377, 10375, 10373, 10372, 10297]
m2_gmt_time = [
    "08:26:30",
    "08:20:40",
    "07:25:00",
    "07:11:48",
    "08:39:30",
    "08:46:00",
    "08:51:00",
    "08:56:00",
]
m2_inst_time = [
    "08:26:38",
    "08:20:34",
    "07:27:17",
    "07:12:16",
    "08:39:45",
    "08:46:37",
    "08:51:51",
    "08:56:33",
]


# %%
def generate_timestamp(time):
    if time == "N/A":
        return np.datetime64("NaT")
    else:
        datetimestr = f"{date} {time}"
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

gmtd = {r: g for r, g in zip(rbr_sn, gmt)}
instd = {r: g for r, g in zip(rbr_sn, inst)}

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
{
    "units": "°C",
    "long_name": "temperature",
    "SN": 207286,
    "model": "RBRsolo³",
    "firmware version": "1.054",
    "file": "207286_20211007_1555.rsk",
    "time drift in ms": -686937060429,
    "download time": "N/A",
    "sample size": 17711599,
    "sampling period in s": 0.5,
    "time offset applied": 0,
}

# %%
tmp = xr.open_dataarray(
    "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/RBRSolo/proc/201912_time_corrected.nc"
)


# %%
def correct_time(rbr):
    print("SN", rbr)
    file = data_out.joinpath(f"time_not_corrected_{rbr:06}_mavs_shipproc.nc")

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
    offset = np.timedelta64(insttime - utctime, "ms").astype("int") * scale_factor

    print(offset / 1e3, "s")

    t.attrs["time drift in ms"] = offset
    t.attrs["nvalues"] = t.size

    old_time = t.time.copy()
    time_offset_linspace = np.linspace(
        0, t.attrs["time drift in ms"], t.attrs["nvalues"]
    )
    # convert to numpy timedelta64
    # this format can't handle non-integers, so we switch to nanoseconds
    time_offset = [
        np.timedelta64(int(np.round(ti * 1e6)), "ns") for ti in time_offset_linspace
    ]
    new_time = old_time - time_offset

    t["time"] = new_time
    t.attrs["time offset applied"] = 1
    sample_size = len(tmp)
    t.attrs["sample size"] = sample_size
    sampling_period = tmp.time.diff(dim="time").median().data
    sampling_period = np.round(sampling_period.astype(int) / 1e9, 2)
    t.attrs["sampling period in s"] = sampling_period
    t.attrs["units"] = "°C"
    t.attrs["long_name"] = "temperature"
    t.attrs["SN"] = (rbr,)
    t.attrs["model"] = "RBRsolo"
    t.attrs["firmware version"] = "N/A"

    # save
    savename = data_out.joinpath(f"{rbr:06}_time_corrected.nc")
    t.to_netcdf(savename, mode="w")
    t.close()


# %%
for sn in tqdm(rbrsn[5:]):
    correct_time(sn)

# %%
