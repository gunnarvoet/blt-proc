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

# %% [markdown] heading_collapsed=true
# ### Imports

# %% hidden=true
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


# %% hidden=true
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # BLT SBE56 Processing v1

# %% [markdown]
# Processing parameters

# %%
mooring_dir = Path("/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/")

sbe_dir = mooring_dir.joinpath("MAVS/SBE56")
data_raw = sbe_dir.joinpath("raw")
data_out = sbe_dir.joinpath("proc")
figure_out = sbe_dir.joinpath("fig")

# Create directories if needed
for d in [data_out, figure_out]:
    d.mkdir(exist_ok=True)


# %% [markdown]
# ## Get file name

# %%
def get_file_name(sn, data_raw, extension='csv'):
    files = list(data_raw.glob(f'SBE056{sn:05}*.{extension}'))
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        raise OSError(f'more than one file for SN{sn} in {data_raw}')


# %%
sn = 6438
file = get_file_name(sn, data_raw)
print(file)


# %% [markdown]
# ## set up processing database

# %%
def generate_proc_info():
    cal_file = 'blt1_sbe56_time_offsets.csv'
    proc_info = pd.read_csv(
        cal_file,
        engine="python",
        header=0,
        delim_whitespace=False,
        parse_dates={
            "utc": [2, 3],
            "inst": [4, 5],
            "cal1": [6, 7],
            "cal2": [8, 9],
        },
        index_col="SN",
    )
    proc_info.utc = pd.to_datetime(proc_info.utc, errors='coerce', utc=True)
    proc_info.inst = pd.to_datetime(proc_info.inst, errors='coerce', utc=True)
    proc_info.cal1 = pd.to_datetime(proc_info.cal1, errors='coerce', utc=True)
    proc_info.cal2 = pd.to_datetime(proc_info.cal2, errors='coerce', utc=True)

    n = proc_info.index.shape[0]

    proc_info = proc_info.assign(processed=np.tile(False, n))
    proc_info = proc_info.assign(raw_data_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(figure_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(comment=np.tile("ok", n))

    proc_info = proc_info.sort_index()
    return proc_info


# %% [markdown]
# Mark existing raw files in the database

# %%
def update_proc_info():
    for g, v in proc_info.groupby('SN'):
        try:
            f = get_file_name(g, data_raw=data_raw, extension='csv')
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


# %%
def add_comment(sn, comment, proc_info):
    if type(sn) is float:
        sn = [sn]
    for si in sn:
        if proc_info.comment.at[si] == "ok":
            proc_info.comment.at[si] = comment
        if comment not in proc_info.comment.at[si]:
            addstr = "\\\\"
            proc_info.comment.at[si] = proc_info.comment.at[si] + addstr + comment


# %%
proc_info = generate_proc_info()
update_proc_info()

# %%
proc_info.head()


# %%
def show_info():
    """Show instruments that are not ok"""
    tmp = proc_info[proc_info.comment != 'ok'].drop(['utc', 'inst', 'cal1', 'cal2'], axis=1)
    from IPython.display import display
    display(tmp)


# %% [markdown]
# ## helper functions

# %%
def runproc(sn):
    utctime = proc_info.loc[sn]['utc']
    insttime = proc_info.loc[sn]['inst']
    cals = get_cal_time(sn)
    file = get_file_name(sn, data_raw, extension='csv')
    print(file.name)
    t = sbe.sbe56.proc(
            file,
            time_instrument=insttime,
            time_utc=utctime,
            data_out=data_out,
            figure_out=figure_out,
            cal_time=cals,
            show_plot=True,
        )


# %%
def get_cal_time(sn):
    cal1 = proc_info.loc[sn]['cal1'].to_datetime64()
    cal2 = proc_info.loc[sn]['cal2'].to_datetime64()
    return cal1, cal2


# %%
def plot(sn):
    data = load_nc(sn)
    cals = get_cal_time(sn)
    sbe.sbe56.plot(data, figure_out=figure_out, cal_time=cals)


# %%
def proc_next():
    update_proc_info()
    sn = proc_info.index.where(~proc_info.processed & proc_info.raw_data_exists).dropna()[0]
    sn = int(sn)
    runproc(sn)


# %%
def plot_next():
    update_proc_info()
    sn = proc_info.index.where(~proc_info.figure_exists & proc_info.raw_data_exists).dropna()[0]
    sn = int(sn)
    plot(sn)


# %%
def load_nc(sn):
    file = get_file_name(sn, data_raw=data_out, extension='nc')
    data = xr.open_dataarray(file, engine='netcdf4')
#     data.attrs["sampling period in s"] = data.attrs["sampling period"]
    return data


# %%
def load_all_nc():
    sn = proc_info[proc_info.processed].index.to_numpy()
    allnc = [load_nc(si) for si in sn]
    return allnc


# %% [markdown]
# ## process next

# %%
while(1):
    proc_next()

# %% [markdown]
# ## SN455

# %% [markdown]
# The clock resets shortly after deployment. Can we recover the time series? We do have the two clock cals in the time series, so maybe can use these? Try and then compare to a working time series from nearby.

# %%
t = load_nc(455)

# %%
tt = t.where(t.time<np.datetime64('2010'), drop=True)

# %% [markdown]
# Plot the intial clock cal

# %%
ts = slice('2000-01-02 06:10:00', '2000-01-02 06:10:20')
tt.sel(time=ts).plot(marker='.')
time1 = np.datetime64('2000-01-02 06:10:11')
plt.vlines(time1, 15, 40, color='purple')

# %%
cal1 = proc_info.loc[455].cal1.to_datetime64()
cal2 = proc_info.loc[455].cal2.to_datetime64()

# %%
offset1 = cal1 - time1
print(offset1)

# %%
tnew = tt.copy()

# %%
tnew['time'] = tt.time + offset1

# %%
tnew.plot()

# %% [markdown]
# Now look at the second clock cal. Seems like we have a 7s time drift that we can adjust for.

# %%
drift = np.timedelta64(7, 's')
ts2 = slice(cal2-np.timedelta64(10, 's'), cal2+np.timedelta64(10, 's'))
tnew.sel(time=ts2).plot(marker='.')
plt.vlines(cal2, 15, 40, color='purple')
plt.vlines(cal2+drift, 15, 40, color='red');

# %%
tnew.time.diff(dim='time')/1e6

# %% [markdown]
# Reset a few fields in the metadata.

# %%
tnew.attrs["sampling period in s"] = (
    tnew.time[:100].diff(dim="time").median().data.astype("timedelta64[ns]").astype(int)
    / 1e9
)

tnew.attrs["time offset applied"] = 0

tnew.attrs["sample size"] = len(tnew)

# %% [markdown]
# Apply the time offset.

# %%
tnew = sbe.sbe56.time_offset(tnew, None, None, drift)

# %%
ts2 = slice(cal2-np.timedelta64(10, 's'), cal2+np.timedelta64(10, 's'))
tnew.sel(time=ts2).plot(marker='.')
plt.vlines(cal2, 15, 40, color='purple');

# %% [markdown]
# Remove the originally processed netcdf file and save the adjusted time series.

# %%
savename = Path('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/SBE56/proc/SBE05600455_2021-10-07.nc')

# %%
tnew.to_netcdf(savename, mode='w')

# %% [markdown]
# Make a new plot.

# %%
cals = get_cal_time(455)
sbe.sbe56.plot(tnew, figure_out=figure_out, cal_time=cals)

# %% [markdown]
# Add a comment to `proc_info`.

# %%
sn = [455]
comment = 'time vector adjusted based on clock calibrations'
add_comment(sn, comment, proc_info)

# %%
update_proc_info()

# %% [markdown]
# ## load all time series

# %%
allnc = load_all_nc()


# %% [markdown]
# ## time series termination

# %% [markdown]
# Only one of the thermistors really ends too early (6432):

# %%
def find_last_time_stamp(thermistor):
    return thermistor.time.isel(time=-1).data


# %%
last_times = np.array([find_last_time_stamp(thi) for thi in allnc])
last_times[last_times<np.datetime64('2021-10-04 00:00:00')]

# %%
sn = [6432]
comment = 'terminates 2021-08-02'
add_comment(sn, comment, proc_info)


# %% [markdown]
# Find the time for each thermistor when the time series ends or starts dropping out too much. Some thermistor have shorter gaps in between.

# %%
def find_gaps(thermistor):
    td = thermistor.time.diff(dim="time")
    dt = np.timedelta64(int(thermistor.attrs["sampling period in s"]) * 1000, "ms")
    tdi = td.where(
        (td > dt + np.timedelta64(500, "ms")) | (td < dt - np.timedelta64(500, "ms")),
        drop=True,
    )
    return tdi


# %%
gaps = {}
for th in allnc:
    gaps[th.attrs['SN'][-4:]] = find_gaps(th)

# %% [markdown]
# List all thermistors that have gaps. Note that the time stamp of the gap is when the instrument comes back online.

# %%
for sn, g in gaps.items():
    if len(g) > 0:
        print(sn, g.time.data-g.data)

# %%
for sn, g in gaps.items():
    if len(g) > 0:
        print(sn, g[0].time.data, np.timedelta64(g[0].data, 'm'))
        print(sn, g[0].time.data-g[0].data)

# %% [markdown]
# SNs with short intermittent gaps (all other time series basically end at the first gap):

# %%
snc = [418, 916, 5462, 6420, 6431, 6446]


# %%
def print_gaps(sn):
    g = gaps[f'{sn:04}']
    for t, d in g.groupby('time'):
        if t<np.datetime64('2021-10-04'):
            print(sn, t, np.timedelta64(d.data, 's'))


# %%
[print_gaps(sni) for sni in snc]

# %% [markdown]
# From the analysis above it seems like we can differentiate between gaps less than one hour (longest intermittent gap is 1540s) and gaps longer than that marking the end of the time series. We can thus loop through and find the first gap that is longer than one hour. While doing this, we can also count the shorter gaps.

# %%
n_short_gaps = {}
end_of_time_series = {}
for sn, g in gaps.items():
    if len(g) > 0:
        short_gaps = 0
        early_end = 0
        for time, gg in g.groupby('time'):
            print(sn, gv.time.convert_units(gg.data, 's'), time-gg.data)
            if gg.data < np.timedelta64(1, 'h'):
                print('short')
                short_gaps += 1
            else:
                end_of_time_series[sn] = time - gg.data
                break
        n_short_gaps[sn] = short_gaps

# %%
for sni, ti in end_of_time_series.items():
    tstr = np.datetime_as_string(ti, unit='D')
    snlist = [int(sni)]
    comment = f'terminates {tstr}'
    add_comment(snlist, comment, proc_info)
    print(comment)

# %%
n_short_gaps

# %%
for sni, n in n_short_gaps.items():
    if n>0:
        snlist = [int(sni)]
        if n>1:
            comment = f'has {n} short gaps (<1h)'
        else:
            comment = f'has 1 short gap (<1h)'
        add_comment(snlist, comment, proc_info)
        print(comment)

# %% [markdown]
# Maybe we need to interpolate over some smaller gaps? Or just leave the smaller gaps? They are too long to interpolate. Good to take note though that they are there.

# %% [markdown] heading_collapsed=true
# ## no time verification at the end

# %% [markdown] hidden=true
# Just make a note of the thermistors that do not have the warm water dip at the end because they were not measuring anymore. Most of them still had the clock running so should be fine.

# %% hidden=true
no_time_check = [392, 425, 6418, 6432]

# %% hidden=true
for sni in no_time_check:
    snlist = [int(sni)]
    comment = f'no final warm water clock cal'
    add_comment(snlist, comment, proc_info)

# %% [markdown] heading_collapsed=true
# ## no data

# %% [markdown] hidden=true
# Add a comment if no data file exists, just to get rid of the ok mark.

# %% hidden=true
sn_no_data = proc_info[~proc_info.raw_data_exists].index.to_numpy()

# %% hidden=true
for sni in sn_no_data:
    snlist = [int(sni)]
    comment = f'no data'
    add_comment(snlist, comment, proc_info)

# %% [markdown]
# ## problems

# %% [markdown]
# Show all instruments that have processing notes.

# %%
show_info()

# %% [markdown]
# ## remove nan in notes

# %%
for sni, note in proc_info.Notes.groupby('SN'):
    if type(note.values[0]) == float:
        proc_info.Notes.at[sni] = ''

# %% [markdown]
# ## save latex table

# %%
proc_info.to_latex(
    "/Users/gunnar/Projects/blt/proc/doc/sbe56_table.tex",
    columns=[
        "Notes",
        "comment",
    ],
    header=["Download Notes", "Processing Notes"],
    longtable=True,
    column_format="p{1cm}p{7cm}p{7cm}",
    multirow=False,
    caption="SBE56 download and processing notes.",
    label="tab:sbe56",
)

# %% [markdown]
# ## cut time series

# %% [markdown]
# We want only the time at the bottom until the last good data point. Save to a level 1 directory if we treat the initially processed netcdf as processed level 0.

# %% [markdown]
# Also, apply the CTD calibration offsets to the level 1 data.

# %%
mavs1_cut_beg = np.datetime64('2021-07-06 14:30:00')
mavs1_cut_end = np.datetime64('2021-10-05 09:30:00')

mavs2_cut_beg = np.datetime64('2021-07-07 13:50:00')
mavs2_cut_end = np.datetime64('2021-10-04 13:00:00')

# %% [markdown] heading_collapsed=true
# ### MAVS1 times

# %% [markdown] hidden=true
# 395 is on MAVS1 and has a good time series.

# %% hidden=true
tmp = load_nc(395)

# %% hidden=true
tmp.sel(time=slice('2021-07-06 14:00:00', '2021-07-06 15:00:00')).plot();

# %% hidden=true
tmp.sel(time=slice('2021-10-05 00:00:00', '2021-10-05 10:00:00')).plot();

# %% hidden=true
tmp.sel(time=slice(mavs1_cut_beg, mavs1_cut_end)).plot();

# %% [markdown] heading_collapsed=true
# ### MAVS2 times

# %% [markdown] hidden=true
# For MAVS2 load 458.

# %% hidden=true
tmp = load_nc(458)

# %% hidden=true
tmp.sel(time=slice('2021-10-04 13:00:00', '2021-10-04 14:00:00')).plot();

# %% hidden=true
tmp.sel(time=slice('2021-07-07 13:30:00', '2021-07-07 14:00:00')).plot();

# %% hidden=true
tmp.sel(time=slice(mavs2_cut_beg, mavs2_cut_end)).plot();

# %% [markdown]
# ### cut & calibrate

# %%
cal = xr.open_dataarray('blt1_sbe56_ctd_cal_offsets.nc')

# %%
cal.sel(sn=376)

# %% [markdown]
# Cals are positive. Thermistors were colder than CTD so we need to add the cals.

# %%
mavs1_info_csv = '/Users/gunnar/Projects/blt/moorings/thermistors/blt_mavs1_thermistors_only.csv'
mavs2_info_csv = '/Users/gunnar/Projects/blt/moorings/thermistors/blt_mavs2_thermistors_only.csv'

# %%
mavs1_info = pd.read_csv(mavs1_info_csv, sep=",", header=0, index_col="SN")
mavs2_info = pd.read_csv(mavs2_info_csv, sep=",", header=0, index_col="SN")

# %%
mavs1_sbe = mavs1_info.query('Type=="SBE56"')
mavs2_sbe = mavs2_info.query('Type=="SBE56"')


# %%
def cut_thermistor(sn, start, end):
    th = load_nc(sn)
    # add cal (we don't have one for 455)
    try:
        sensorcal = cal.sel(sn=sn)
        th = th + sensorcal.data
    except:
        print(f'no cal for {sn}')
    out = th.sel(time=slice(start, end), drop=True)
    level2 = Path('/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/MAVS/SBE56/proc_L1')
    outname = level2.joinpath(f'blt1_sbe56_{sn:04}.nc')
    out.to_netcdf(outname, mode='w')
    return out


# %%
sn = mavs1_sbe.index.to_numpy()
for sni in sn:
    test = cut_thermistor(sni, mavs1_cut_beg, mavs1_cut_end)

# %%
sn = mavs2_sbe.index.to_numpy()
for sni in sn:
    test = cut_thermistor(sni, mavs2_cut_beg, mavs2_cut_end)

# %% [markdown]
# ## plot spectra

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
