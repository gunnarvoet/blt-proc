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

# %% [markdown] heading_collapsed=true
# #### Imports

# %% hidden=true
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xarray as xr
from pathlib import Path

import gvpy as gv

# %config InlineBackend.figure_format = 'retina'
# %reload_ext autoreload
# %autoreload 2

# %% [markdown]
# # BLT Trilateration

# %% [markdown]
# Moved all trilateration functions to `gvpy.trilaterate`.
#
# Running trilateration for all BLT moorings here to produce netcdf and ascii file with locations and depths.
#
# Still need to
# - include sADCP data in the analysis to see if we can do a better job with positioning in the future

# %% [markdown] heading_collapsed=true
# ### Navigation data

# %% [markdown] hidden=true
# Read met data for locations of BLT trilateration surveys.

# %% hidden=true
met2 = xr.open_dataset(Path('/Users/gunnar/Projects/blt/cruises/blt2/py/blt2_met.nc'))
met3 = xr.open_dataset(Path('/Users/gunnar/Projects/blt/cruises/blt3/py/blt3_met.nc'))
met = xr.concat([met2, met3], dim='time')

# %% hidden=true
nav = xr.Dataset(data_vars=dict(lon=(('time'), met.long.data), lat=(('time'), met.lat.data)), coords=dict(time=met.time.data))

# %% hidden=true
met2.close()
met3.close()
met.close()

# %% hidden=true
nav = nav.where(~np.isnat(nav.time), drop=True)

# %% hidden=true
navsel = nav.sel(time='2022-08-04')

# %% hidden=true
navsel.lon.plot()

# %% [markdown] heading_collapsed=true
# ### Bathymetry

# %% hidden=true
topo = xr.open_dataarray(Path('/Users/gunnar/Projects/blt/proc/mb/blt/final/blt_canyon_mb_qc.nc'))

# %% hidden=true
topo_extent = (
    topo.lon.min().data,
    topo.lon.max().data,
    topo.lat.max().data,
    topo.lat.min().data,
)

# %% [markdown] heading_collapsed=true
# ### BLT1

# %% [markdown] heading_collapsed=true hidden=true
# #### MP1

# %% hidden=true
lat = np.array([54.24411189, 54.23283283, 54.23704613])
lon = np.array([-11.953817, -11.95783703, -11.93493788])
pos = [(loni, lati) for loni, lati in zip(lon, lat)]
ranges = np.array([2146, 2204, 2221])

plan_lon, plan_lat = -11-56.958/60, 54 + 14.334/60
bottom_depth = 2034
mp1 = gv.trilaterate.Trilateration(
    "MP1",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    topo=topo,
)
for p, d in zip(pos, ranges):
    mp1.add_ranges(distances=d, pos=p)

# %% hidden=true
mp1.trilaterate(i=0)

# %% hidden=true
mp1.plot_results(pmlat=0.005);

# %% [markdown] heading_collapsed=true hidden=true
# #### MP2

# %% [markdown] hidden=true
# The trilateration survey points for MP2 and MAVS2 were chosen a little unfortunate. We should calculate the point from the two-point solutions instead.
#
# How about checking whether the three-point solution is contained within the triangle made up by the two-point solutions?

# %% hidden=true
lat = np.array([54.209717, 54.194503, 54.179905])
lon = np.array([-11.839856, -11.854456, -11.869637])
pos = [(loni, lati) for loni, lati in zip(lon, lat)]
ranges = np.array([2835, 2269, 3141])

plan_lon, plan_lat = -11.873, 54.203
bottom_depth = 1666
mp2 = gv.trilaterate.Trilateration(
    "MP2",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    topo=topo,
)
for p, d in zip(pos, ranges):
    mp2.add_ranges(distances=d, pos=p)

# %% hidden=true
mp2.trilaterate(i=0)

# %% hidden=true
mp2.plot_locations();

# %% hidden=true
mp2.result

# %% hidden=true
mp2.plot_results(pmlat=0.01);

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS1

# %% hidden=true
lat = np.array([54.179456, 54.199803, 54.199601])
lon = np.array([-11.859335, -11.879496, -11.839329])
pos = [(loni, lati) for loni, lati in zip(lon, lat)]
ranges = np.array([2573, 1987, 2193])

plan_lon, plan_lat = -11-51.630/60, 54 + 11.868/60
bottom_depth = 1612
mavs1 = gv.trilaterate.Trilateration(
    "MAVS1",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    topo=topo,
)
for p, d in zip(pos, ranges):
    mavs1 .add_ranges(distances=d, pos=p)

# %% hidden=true
mavs1.trilaterate(i=0)

# %% hidden=true
mavs1.plot_results(pmlat=0.01);

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS2

# %% hidden=true
lat = np.array([54.209717, 54.194503, 54.179905])
lon = np.array([-11.839856, -11.854456, -11.869637])
pos = [(loni, lati) for loni, lati in zip(lon, lat)]
ranges = np.array([3397, 2124, 2305])

plan_lon, plan_lat = -11.843, 54.183
bottom_depth = 1461
mavs2 = gv.trilaterate.Trilateration(
    "MAVS2",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    topo=topo,
)
for p, d in zip(pos, ranges):
    mavs2 .add_ranges(distances=d, pos=p)

# %% hidden=true
mavs2.trilaterate(i=0)

# %% hidden=true
mavs2.plot_results();

# %% hidden=true
mavs2.plot_locations();

# %% [markdown] heading_collapsed=true hidden=true
# #### TCHAIN

# %% hidden=true
lat = np.array([54.179456, 54.199803, 54.199601])
lon = np.array([-11.859335, -11.879496, -11.839329])
pos = [(loni, lati) for loni, lati in zip(lon, lat)]
ranges = np.array([1984, 2558, 2022])

plan_lon, plan_lat = -11-51.120/60, 54 + 11.430/60
bottom_depth = 1525
tchain = gv.trilaterate.Trilateration(
    "TCHAIN",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    topo=topo,
)
for p, d in zip(pos, ranges):
    tchain.add_ranges(distances=d, pos=p)

# %% hidden=true
tchain.trilaterate(i=0)

# %% hidden=true
tchain.plot_results(pmlat=0.005);

# %% [markdown] heading_collapsed=true
# ### BLT2

# %% [markdown] heading_collapsed=true hidden=true
# #### MP3

# %% [markdown] hidden=true
# ```
# 2021-10-22 10:09:52
# MP3 Mooring deployed
# lat 54.185246
# lon -11.848392
# heading deg 269.5
# wind deg 344
# wind speed m/s 9.330000
#
# 2021-10-22 08:43:37
# Commence mooring deployment MP3
# lat 54.185808
# lon -11.827157
# heading deg 268.600006
# wind deg 339
# wind speed m/s 10.350000
# ```

# %% hidden=true
trilat_mp3 = dict()

trilat_mp3["point1"] = dict(
    times=["2021-10-22 21:02:55", "2021-10-22 21:03:11", "2021-10-22 21:03:20"],
    ranges=[2288, 2292, 2295],
)

# note: not including the first sounding at this location, must have included some echos
trilat_mp3["point2"] = dict(
    times=[
        "2021-10-22 21:35:25",
        "2021-10-22 21:35:38",
        "2021-10-22 21:35:55",
        "2021-10-22 21:36:05",
    ],
    ranges=[2482, 2488, 2495, 2498],
)

trilat_mp3["point3"] = dict(
    times=["2021-10-22 22:10:48", "2021-10-22 22:11:05", "2021-10-22 22:11:16"],
    ranges=[2331, 2329, 2328],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 50.801 / 60, 54 + 11.118 / 60
bottom_depth = 1491
mp3 = gv.trilaterate.Trilateration(
    "MP3",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time='2021-10-22 10:09:52',
)
for p, point  in trilat_mp3.items():
    mp3.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
mp3.trilaterate(0)

# %% hidden=true
ax = mp3.plot_results(pmlat=0.005)
# ax.plot(mp3.drop_approach.lon, mp3.drop_approach.lat, transform=ccrs.PlateCarree(), color='green')
# ax.plot(mp3.drop_pos.lon, mp3.drop_pos.lat, transform=ccrs.PlateCarree(), marker='d', color='orange')

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS3

# %% [markdown] hidden=true
# ```
# 2021-10-22 20:15:45
# MAVS 3 mooring released from ship
# lat 54.18178
# lon -11.851208
# heading deg 198.5
# wind dir 332
# wind speed m/s 10.710000
#
# 2021-10-22 16:31:54
# Commenced MAVS3 deployment
# lat 54.206549
# lon -11.838565
# heading deg 200.100006
# wind deg 359
# wind speed m/s 9.620000
# ```

# %% hidden=true
trilat_mavs3 = dict()

trilat_mavs3["point1"] = dict(
    times=["2021-10-22 21:06:00", "2021-10-22 21:06:15", "2021-10-22 21:06:25"],
    ranges=[2017, 2016, 2016],
)

trilat_mavs3["point2"] = dict(
    times=[
        "2021-10-22 21:38:18",
        "2021-10-22 21:38:30",
        "2021-10-22 21:38:42",
        "2021-10-22 21:38:55",
    ],
    ranges=[2755, 2749, 2744, 2738],
)

trilat_mavs3["point3"] = dict(
    times=["2021-10-22 22:13:28", "2021-10-22 22:13:40", "2021-10-22 22:14:00"],
    ranges=[2289, 2285, 2284],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 51.041 / 60, 54 + 10.968 / 60
bottom_depth = 1433
mavs3 = gv.trilaterate.Trilateration(
    "MAVS3",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time='2021-10-22 20:15:45',
)
for p, point  in trilat_mavs3.items():
    mavs3.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
mavs3.trilaterate(0)

# %% hidden=true
ax = mavs3.plot_results(pmlat=0.002);

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS4

# %% [markdown] hidden=true
# ```
# 2021-10-22 15:55:44
# mooring deployed
# lat 54.197629
# lon -11.843911
# heading 10.900000
# wind dir 206
# wind speed m/s 6.120000
# Note: This log entry was taken later than anchor drop. Guessing anchor drop time based on distance between target location and drop location that should be nominally 50m.
#
# 2021-10-22 11:35:07
# Commence MAVS4 deployment, 2nm downwind
# lat 54.198422
# lon -11.790249
# heading 253.199997
# wind dir 332
# wind speed m/s 7.570000
# ```

# %% hidden=true
trilat_mavs4 = dict()

trilat_mavs4["point1"] = dict(
    times=['2021-10-22 21:09:05', '2021-10-22 21:09:16', '2021-10-22 21:09:25'],
    ranges=[2470, 2469, 2468],
)

trilat_mavs4["point2"] = dict(
    times=['2021-10-22 21:41:10', '2021-10-22 21:41:20', '2021-10-22 21:41:40', '2021-10-22 21:41:50'],
    ranges=[2153, 2146, 2142, 2141],
)

trilat_mavs4["point3"] = dict(
    times=['2021-10-22 22:16:08', '2021-10-22 22:16:25', '2021-10-22 22:16:45', '2021-10-22 22:17:10', '2021-10-22 22:17:25', '2021-10-22 22:18:00'],
    ranges=[2365, 2379, 2394, 2414, 2433, 2448],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 50.591 / 60, 54 + 11.268 / 60
bottom_depth = 1445
mavs4 = gv.trilaterate.Trilateration(
    "MAVS4",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time="2021-10-22 15:40:00",
)
for p, point  in trilat_mavs4.items():
    mavs4.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
mavs4.trilaterate(0)

# %% hidden=true
ax = mavs4.plot_results(pmlat=0.001)

# %% [markdown] heading_collapsed=true
# ### BLT3

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS3 prior to recovery

# %% hidden=true
trilat_mavs3a = dict()

trilat_mavs3a["point1"] = dict(
    times=["2022-08-04 12:58:30", "2022-08-04 12:58:50", "2022-08-04 12:59:02"],
    ranges=[1579, 1580, 1581],
)

trilat_mavs3a["point2"] = dict(
    times=[
        "2022-08-04 13:17:25",
        "2022-08-04 13:17:36",
    ],
    ranges=[1473, 1472],
)

trilat_mavs3a["point3"] = dict(
    times=["2022-08-04 13:36:13", "2022-08-04 13:36:22", "2022-08-04 13:36:40"],
    ranges=[1540, 1542, 1542],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 51.041 / 60, 54 + 10.968 / 60
bottom_depth = 1433
mavs3a = gv.trilaterate.Trilateration(
    "MAVS3a",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time='2021-10-22 20:15:45',
)
for p, point  in trilat_mavs3a.items():
    mavs3a.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
mavs3a.trilaterate(0)

# %% hidden=true
ax = mavs3a.plot_results(pmlat=0.002);

# %% hidden=true
ax = mavs3.plot_results(pmlat=0.002);

# %% [markdown] heading_collapsed=true hidden=true
# #### MAVS4 prior to recovery

# %% hidden=true
trilat_mavs4a = dict()

trilat_mavs4a["point1"] = dict(
    times=['2022-08-04 13:02:02', '2022-08-04 13:02:32'],
    ranges=[1553, 1553],
)

trilat_mavs4a["point2"] = dict(
    times=['2022-08-04 13:19:37', '2022-08-04 13:19:50', '2022-08-04 13:20:03'],
    ranges=[1477, 1473, 1478],
)

trilat_mavs4a["point3"] = dict(
    times=['2022-08-04 13:30:41', '2022-08-04 13:30:50'],
    ranges=[1546, 1545],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 50.591 / 60, 54 + 11.268 / 60
bottom_depth = 1445
mavs4a = gv.trilaterate.Trilateration(
    "MAVS4a",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time="2021-10-22 15:40:00",
)
for p, point  in trilat_mavs4a.items():
    mavs4a.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
mavs4a.trilaterate(0)

# %% hidden=true
ax = mavs4a.plot_results(pmlat=0.001)

# %% hidden=true
ax = mavs4.plot_results(pmlat=0.001)

# %% [markdown] heading_collapsed=true hidden=true
# #### G1

# %% hidden=true
trilat_g1 = dict()

trilat_g1["point1"] = dict(
    times=["2022-08-04 19:37:40", "2022-08-04 19:38:00", "2022-08-04 19:38:15"],
    ranges=[1363, 1365, 1365],
)

trilat_g1["point2"] = dict(
    times=[
        "2022-08-04 20:05:20",
        "2022-08-04 20:05:32",
        "2022-08-04 20:05:42",
    ],
    ranges=[1372, 1374, 1374],
)

trilat_g1["point3"] = dict(
    times=["2022-08-04 20:24:35", "2022-08-04 20:24:50", "2022-08-04 20:25:00"],
    ranges=[1352, 1352, 1351],
)

# %% hidden=true
plan_lon, plan_lat = -11 - 49.8 / 60, 54 + 10.3 / 60
bottom_depth = 1274
g1 = gv.trilaterate.Trilateration(
    "G1",
    plan_lon=plan_lon,
    plan_lat=plan_lat,
    bottom_depth=bottom_depth,
    nav=nav,
    topo=topo,
    drop_time='2022-08-04 18:59:25',
)
for p, point  in trilat_g1.items():
    g1.add_ranges(times=point["times"], distances=point["ranges"])

# %% hidden=true
g1.trilaterate(0)

# %% hidden=true
ax = g1.plot_results(pmlat=0.002);

# %% [markdown]
# ### Gather Moorings

# %%
blt = [mp1, mp2, mp3, mavs1, mavs2, mavs3, mavs4, tchain, g1]

# %%
tmp = [mi.to_netcdf() for mi in blt]
m = xr.concat(tmp, dim='mooring')

# %%
m.offset

# %%
[mi.print_result() for mi in blt];

# %% [markdown]
# Save results

# %%
out_txt = Path('/Users/gunnar/Projects/blt/moorings/blt_mooring_locations.txt')

# %%
with open(out_txt, "w") as text_file:
    for g, mi in m.groupby('mooring'):
        lon, lat = gv.ocean.lonlatstr(mi.lon_actual.item(), mi.lat_actual.item(), )
        print(f'{mi.mooring.item():8s}', file=text_file)
        print(f'{lat:15} {lon:15}', file=text_file)
        print(f'{mi.lon_actual.item():4.3f}, {mi.lat_actual.item():4.3f}', file=text_file)
        print(f'{mi.depth_actual.item():4.0f}m\n', file=text_file)

# %%
m.to_netcdf('blt_trilateration_results.nc')

# %%
m.to_netcdf('/Users/gunnar/Projects/blt/moorings/blt_mooring_locations.nc')
