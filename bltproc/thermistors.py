#!/usr/bin/env python
"""Module for processing moored BLT thermistors."""

from pathlib import Path
import os
import gvpy as gv
import numpy as np
import xarray as xr
import pandas as pd
import gsw
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sbemoored as sbe
from IPython.display import display
import shutil


def test():
    print("hello world!")


def mavs_cut_times(mavsid=1):
    if mavsid == 1:
        cut_beg = np.datetime64("2021-07-06 14:30:00")
        cut_end = np.datetime64("2021-10-05 09:30:00")
    elif mavsid == 2:
        cut_beg = np.datetime64("2021-07-07 13:50:00")
        cut_end = np.datetime64("2021-10-04 13:00:00")
    return cut_beg, cut_end


def rbr_save_development_chunks(l0_data, chunk_dir):
    # save short chunks of data for development
    time_span = slice("2021-07-31 00:00:00", "2021-08-04 00:00:00")
    files = sorted(l0_data.glob("*.nc"))

    def save_short_chunk(file, ctd_time):
        r = xr.open_dataarray(file)
        outname = f'{file.stem[:file.stem.find("_")]}_proc_dev.nc'
        outpath = chunk_dir.joinpath(outname)
        r.sel(time=ctd_time).to_netcdf(outpath)
        r.close()

    for file in files:
        save_short_chunk(file, time_span)


def rbr_save_ctd_cal_time_series(l0_data, ctd_time, save_dir):
    """Save thermistor time series for time of CTD calibration cast."""
    files = sorted(l0_data.glob("*.nc"))

    def save_ctd_cal(file, ctd_time):
        r = xr.open_dataarray(file)
        datestr = ctd_time.start[:10].replace("-", "")
        outname = f'{file.stem[:file.stem.find("_")]}_ctd_cal_cast_{datestr}.nc'
        outpath = save_dir.joinpath(outname)
        r.sel(time=ctd_time).to_netcdf(outpath)
        r.close()

    for file in files:
        save_ctd_cal(file, ctd_time)


def rbr_load_cals(cal_dir):
    calfiles = sorted(cal_dir.glob("*.nc"))
    cals = [xr.open_dataarray(calfile) for calfile in calfiles]
    sns = [ci.attrs["SN"] for ci in cals]

    # pick one time for all of them and interpolate
    time = cals[0].time.copy()
    calsi = [ci.interp_like(time) for ci in cals]
    c = xr.concat(calsi, dim="n")
    c["sn"] = (("n"), sns)
    return c


def plot_zoom(ax, ts, rbr, ctd, add_legend=False):
    t = rbr.sel(time=ts)
    t.plot(
        hue="n", add_legend=False, linewidth=0.75, color="k", alpha=0.3, ax=ax
    )
    ctd.t1.sel(time=ts).plot(color="C0", linewidth=1, label="CTD 1", ax=ax)
    ctd.t2.sel(time=ts).plot(color="C4", linewidth=1, label="CTD 2", ax=ax)
    ax.set(xlabel="", ylabel="in-situ temperature [°C]")
    gv.plot.concise_date(ax)
    gv.plot.axstyle(ax, fontsize=9)
    if add_legend:
        ax.legend()


def plot_cal_stop(ts, rbr, ctd, dt=2):
    """Plot CTD calibration stop with zoom."""
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True
    )
    plot_zoom(ax[0], ts, rbr, ctd, add_legend=True)
    tsn = slice(
        np.datetime64(ts.start) - np.timedelta64(dt, "m"),
        np.datetime64(ts.stop) + np.timedelta64(dt, "m"),
    )
    plot_zoom(ax[1], tsn, rbr, ctd)
    mctd = ctd.t1.sel(time=ts).mean()
    ax[1].plot(
        [np.datetime64(ti) for ti in [ts.start, ts.stop]],
        np.tile(mctd - 0.02, 2),
        "r",
    )


def plot_multiple_cal_stop(ts, rbr, ctd, dt=2):
    """Plot multiple CTD calibration stop with zoom."""
    n = len(ts)
    fig, ax = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(10, 3 * n),
        constrained_layout=True,
    )
    leg = True
    ii = 1
    for tsi, axi in zip(ts, ax):
        plot_zoom(axi[0], tsi, rbr, ctd, add_legend=leg)
        tsn = slice(
            np.datetime64(tsi.start) - np.timedelta64(dt, "m"),
            np.datetime64(tsi.stop) + np.timedelta64(dt, "m"),
        )
        gv.plot.annotate_corner(ii, axi[0], background_circle="0.1", col="w")
        plot_zoom(axi[1], tsn, rbr, ctd)
        mctd = ctd.t1.sel(time=tsi).mean()
        axi[1].plot(
            [np.datetime64(ti) for ti in [tsi.start, tsi.stop]],
            np.tile(mctd - 0.02, 2),
            "r",
        )
        leg = False
        ii += 1
    return fig, ax


def rbr_ctd_cal_find_offset(ts, rbr, ctd):
    ctdm1 = ctd.t1.sel(time=ts).mean().data
    ctdm2 = ctd.t2.sel(time=ts).mean().data
    print(
        f"Difference between CTD sensor 1 & 2 mean values: {ctdm1-ctdm2:.4e}°C"
    )
    ctdm = np.mean([ctdm1, ctdm2])
    ab = rbr.sel(time=ts)
    diffmean_1 = (ctdm1 - ab).mean(dim="time")
    diffmean_1.name = "mean1"
    diffstd_1 = (ctdm1 - ab).std(dim="time")
    diffstd_1.name = "std1"
    diffmean_2 = (ctdm2 - ab).mean(dim="time")
    diffmean_2.name = "mean2"
    diffstd_2 = (ctdm2 - ab).std(dim="time")
    diffstd_2.name = "std2"
    diffmean_both = (ctdm - ab).mean(dim="time")
    diffmean_both.name = "meanboth"
    diffstd_both = (ctdm - ab).std(dim="time")
    diffstd_both.name = "stdboth"
    mean = xr.concat([diffmean_both, diffmean_1, diffmean_2], dim="m")
    mean = mean.swap_dims({"n": "sn"})
    mean.name = "mean_diff"
    std = xr.concat([diffstd_both, diffstd_1, diffstd_2], dim="m")
    std = std.swap_dims({"n": "sn"})
    std.name = "std_diff"

    mean_temp = ab.mean(dim="time")
    mean_temp.name = "mean_temp"
    mean_temp = mean_temp.swap_dims({"n": "sn"})

    out = xr.merge([mean, std, mean_temp])
    out.coords["sensor"] = (("m"), ["1", "2", "both"])

    return out


def rbr_ctd_cal_plot_all_sensor_offsets(c, selected_points):
    tmp = c.isel(m=2)
    # tmp = tmp.isel(sn=0)
    fig, ax = gv.plot.quickfig()
    for sn, b in tmp.groupby("sn"):
        gv.plot.axstyle(ax)

        ax.hlines(
            b.isel(calpoint=selected_points).mean_diff.mean() * 1e3,
            0,
            11,
            color="b",
            linewidth=0.7,
        )
        ax.plot(b.mean_temp[:3], (b.mean_diff[:3] * 1e3), "k1", alpha=0.8)
        ax.plot(
            b.mean_temp[3],
            (b.mean_diff[3] * 1e3),
            color="k",
            marker=6,
            linestyle="",
            alpha=0.8,
        )
        ax.plot(
            b.mean_temp[4],
            (b.mean_diff[4] * 1e3),
            color="k",
            marker=7,
            linestyle="",
            alpha=0.8,
        )
        ax.plot(b.mean_temp[5:], (b.mean_diff[5:] * 1e3), "k2", alpha=0.8)
        ax.set(
            ylim=(-10, 10),
            xlim=(2, 11),
            title=f"BLT1 RBR {sn} CTD Cal Offsets",
            xlabel="temperature [°C]",
            ylabel="temperature offset [mK]",
        )
        if sn == 72216:
            ax.set(ylim=(-20, 20))
        gv.plot.png(
            fname=f"{sn:06}_cal_offsets", figdir="cal_offsets", verbose=False
        )
        ax.cla()


def rbr_find_last_time_stamp(thermistor):
    return thermistor.time.isel(time=-1).data


def rbr_load_proc_level0(sn, l0dir):
    files = list(l0dir.glob(f"{sn:06}*.nc"))
    if len(files) == 1:
        return xr.open_dataarray(files[0])
    elif len(files) == 0:
        return None
    else:
        raise OSError(f"more than one file for SN{sn} in {data_raw}")


def rbr_find_gaps(thermistor):
    return find_gaps(thermistor)


def find_gaps(thermistor):
    td = thermistor.time.diff(dim="time")
    dt = np.timedelta64(
        int(thermistor.attrs["sampling period in s"] * 1000), "ms"
    )
    tdi = td.where(
        (td > dt + np.timedelta64(500, "ms"))
        | (td < dt - np.timedelta64(500, "ms")),
        drop=True,
    )
    return tdi


def find_first_long_gap(tdi):
    ti = tdi > np.timedelta64(1, "h")
    if np.any(ti):
        t = tdi.where(ti, drop=True)
        t0 = t.isel(time=0)
    else:
        t0 = np.datetime64("nat")
    return t0


def rbr_find_first_long_gap(tdi):
    t0 = find_first_long_gap(tdi)
    t0["time"] = (t0.time - t0).data
    return


def rbr_apply_ctd_offset(thermistor, sn, ctdcal):
    if sn in ctdcal.sn:
        cal = ctdcal.sel(sn=sn).data
        return thermistor + cal
    else:
        print(f"no cal for {sn}")
        return thermistor


def rbr_blt1_load_mooring_sensor_info(thermistor_info_path):
    return blt1_load_mooring_sensor_info(
        thermistor_info_path, thermistor_type="rbr"
    )


def rbr_cut_and_cal(sn, l0dir, l1dir, thermistor_info_path, ctdcal):
    mavs1_rbr, mavs2_rbr = rbr_blt1_load_mooring_sensor_info(
        thermistor_info_path
    )
    # get deployment times
    if sn in mavs1_rbr.index:
        cut_beg, cut_end = mavs_cut_times(mavsid=1)
        # cut_beg = np.datetime64("2021-07-06 14:30:00")
        # cut_end = np.datetime64("2021-10-05 09:30:00")
    elif sn in mavs2_rbr.index:
        cut_beg, cut_end = mavs_cut_times(mavsid=2)
        # cut_beg = np.datetime64("2021-07-07 13:50:00")
        # cut_end = np.datetime64("2021-10-04 13:00:00")
    else:
        print(f"cannot find SN{sn} in mooring info structure")
    print(f"loading SN{sn}")
    tmp = rbr_load_proc_level0(sn, l0dir)
    attrs = tmp.attrs
    last_time = rbr_find_last_time_stamp(tmp)
    t1 = cut_end
    if last_time < cut_end:
        t1 = last_time
    tdi = rbr_find_gaps(tmp)
    if len(tdi) > 0:
        t = rbr_find_first_long_gap(tdi)
        if ~np.isnat(t):
            if t.time.data < t1:
                t1 = t.time.data

    tmpcut = tmp.where((tmp.time > cut_beg) & (tmp.time < t1), drop=True)

    tmpcal = rbr_apply_ctd_offset(thermistor=tmpcut, sn=sn, ctdcal=ctdcal)
    tmpcal.attrs = attrs
    tmpcal.attrs["sample size"] = len(tmpcal)
    print("saving")
    savename = f"blt1_rbr_{sn:06}.nc"
    tmpcal.to_netcdf(l1dir.joinpath(savename), mode="w")

    return tmpcal


def blt1_load_mooring_sensor_info(thermistor_info_path, thermistor_type="rbr"):
    mavs1_info_csv = thermistor_info_path.joinpath(
        "blt_mavs1_thermistors_only.csv"
    )
    mavs2_info_csv = thermistor_info_path.joinpath(
        "blt_mavs2_thermistors_only.csv"
    )

    mavs1_info = pd.read_csv(mavs1_info_csv, sep=",", header=0, index_col="SN")
    mavs2_info = pd.read_csv(mavs2_info_csv, sep=",", header=0, index_col="SN")

    if thermistor_type == "rbr":
        mavs1 = mavs1_info.query(
            'Type=="Solo" | Type=="Solo Ti" | Type=="MAVS Solo"'
        )
        mavs2 = mavs2_info.query(
            'Type=="Solo" | Type=="Solo Ti" | Type=="MAVS Solo"'
        )
    elif thermistor_type == "sbe":
        mavs1 = mavs1_info.query('Type=="SBE56"')
        mavs2 = mavs2_info.query('Type=="SBE56"')

    return mavs1, mavs2


class blt1_sbe56_proc:
    """Docstring for blt1_sbe56_proc."""

    def __init__(self):
        """TODO: to be defined."""
        self.proc_info = self.generate_proc_info()
        self.load_paths()
        self.create_directories()

    def load_paths(self):
        self.mooring_dir = Path(
            "/Users/gunnar/Projects/blt/data/BLT/Moorings/BLT1/"
        )
        self.doc_dir = Path("/Users/gunnar/Projects/blt/proc/doc/")
        self.sbe_dir = self.mooring_dir.joinpath("MAVS/SBE56")
        self.data_raw = self.sbe_dir.joinpath("raw")
        self.data_out = self.sbe_dir.joinpath("proc_test")
        self.level1_dir = self.sbe_dir.joinpath("proc_L1")
        self.figure_out = self.sbe_dir.joinpath("fig_test")
        self.figure_out_level1 = self.sbe_dir.joinpath("fig_L1")
        self.ctd_cal = Path(
            "/Users/gunnar/Projects/blt/proc/sbe56/blt1/blt1_sbe56_ctd_cal_offsets.nc"
        )
        self.thermistor_info_path = Path(
            "/Users/gunnar/Projects/blt/moorings/thermistors/"
        )

    def generate_proc_info(self):
        cal_file = "blt1_sbe56_time_offsets.csv"
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
        proc_info.utc = pd.to_datetime(proc_info.utc, errors="coerce", utc=True)
        proc_info.inst = pd.to_datetime(
            proc_info.inst, errors="coerce", utc=True
        )
        proc_info.cal1 = pd.to_datetime(
            proc_info.cal1, errors="coerce", utc=True
        )
        proc_info.cal2 = pd.to_datetime(
            proc_info.cal2, errors="coerce", utc=True
        )

        n = proc_info.index.shape[0]

        proc_info = proc_info.assign(processed=np.tile(False, n))
        proc_info = proc_info.assign(raw_data_exists=proc_info.processed.copy())
        proc_info = proc_info.assign(figure_exists=proc_info.processed.copy())
        proc_info = proc_info.assign(comment=np.tile("ok", n))

        proc_info = proc_info.sort_index()
        return proc_info

    def update_proc_info(self):
        for g, v in self.proc_info.groupby("SN"):
            try:
                f = self.get_file_name(g, dir=self.data_raw, extension="csv")
                if f is not None:
                    self.proc_info.raw_data_exists.at[g] = True
                f = self.get_file_name(g, dir=self.data_out, extension="nc")
                if f is not None:
                    self.proc_info.processed.at[g] = True
                f = self.get_file_name(g, dir=self.figure_out, extension="png")
                if f is not None:
                    self.proc_info.figure_exists.at[g] = True
            except:
                pass
        # return proc_info

    def add_comment(self, sn, comment):
        if type(sn) is float:
            sn = [sn]
        for si in sn:
            if self.proc_info.comment.at[si] == "ok":
                self.proc_info.comment.at[si] = comment
            if comment not in self.proc_info.comment.at[si]:
                addstr = "\\\\"
                self.proc_info.comment.at[si] = (
                    self.proc_info.comment.at[si] + addstr + comment
                )

    def create_directories(self):
        # Create directories if needed
        for d in [self.data_out, self.figure_out]:
            d.mkdir(exist_ok=True)

    def get_file_name(self, sn, dir="raw", extension="csv"):
        if dir == "raw":
            dir = self.data_raw
        files = list(dir.glob(f"SBE056{sn:05}*.{extension}"))
        if len(files) == 1:
            return files[0]
        elif len(files) == 0:
            return None
        else:
            raise OSError(f"more than one file for SN{sn} in {dir}")

    def get_cal_time(self, sn):
        cal1 = self.proc_info.loc[sn]["cal1"].to_datetime64()
        cal2 = self.proc_info.loc[sn]["cal2"].to_datetime64()
        return cal1, cal2

    def runproc(self, sn):
        utctime = self.proc_info.loc[sn]["utc"]
        insttime = self.proc_info.loc[sn]["inst"]
        cals = self.get_cal_time(sn)
        show_plot = True
        if sn == 455:
            insttime = None
            utctime = insttime
            show_plot = False
        file = self.get_file_name(sn)
        print(file.name)
        t = sbe.sbe56.proc(
            file,
            time_instrument=insttime,
            time_utc=utctime,
            data_out=self.data_out,
            figure_out=self.figure_out,
            cal_time=cals,
            show_plot=show_plot,
        )

    def proc_next(self):
        self.update_proc_info()
        sn = self.proc_info.index.where(
            ~self.proc_info.processed & self.proc_info.raw_data_exists
        ).dropna()
        if len(sn) > 0:
            sn = sn[0]
            sn = int(sn)
            self.runproc(sn)
        else:
            print("all files processed")

    def load_nc(self, sn):
        file = self.get_file_name(sn, dir=self.data_out, extension="nc")
        data = xr.open_dataarray(file, engine="netcdf4")
        #     data.attrs["sampling period in s"] = data.attrs["sampling period"]
        return data

    def load_all_nc(self):
        """Load all processed thermistor time series."""
        self.update_proc_info()
        sn = self.proc_info[self.proc_info.processed].index.to_numpy()
        self.allnc = [self.load_nc(si) for si in sn]
        # create better serial numbers
        for ti in self.allnc:
            ti.attrs["sn"] = int(ti.attrs["SN"][3:])

    def fix_sn455(self):
        sn = 455
        # see if we have done this already.
        uncorrected_name = "wrong_time_SBE05600455.nc"
        if self.data_out.joinpath(uncorrected_name).exists():
            print("455 already fixed, only adding a comment to proc_info")
        else:
            print("Correcting SN455 time vector.\n")
            filename = self.data_out.joinpath("SBE05600455_2021-10-07.nc")
            t = self.load_nc(sn)
            # Get rid of the little bit of data from before the clock reset.
            # The first clock cal is still after the clock reset so no harm in dropping data here.
            tt = t.where(t.time < np.datetime64("2010"), drop=True)
            # We use the initial clock cal as new time reference.
            time1 = np.datetime64("2000-01-02 06:10:11")
            print(
                "The clock reset before the first clock cal."
                f"We determine {time1} as the initial cal time.\n"
            )
            # Now find the correct time by comparing it to the actual time stamp of the clock cal.
            cal1 = self.proc_info.loc[455].cal1.to_datetime64()
            cal2 = self.proc_info.loc[455].cal2.to_datetime64()
            offset1 = cal1 - time1
            print(
                "We can calculate the offset to the actual first warm water dip:",
                f"{np.timedelta64(offset1, 'h')}.\n",
            )
            # Create a new dataset with a corrected time vector.
            tnew = tt.copy()
            tnew["time"] = tt.time + offset1
            # Now look at the second clock cal. Seems like we have a 7s time drift that we can adjust for.
            drift = np.timedelta64(7, "s")
            ts2 = slice(
                cal2 - np.timedelta64(10, "s"), cal2 + np.timedelta64(10, "s")
            )
            # Reset a few fields in the metadata.
            tnew.attrs["sampling period in s"] = (
                tnew.time[:100]
                .diff(dim="time")
                .median()
                .data.astype("timedelta64[ns]")
                .astype(int)
                / 1e9
            )
            tnew.attrs["time offset applied"] = 0
            tnew.attrs["sample size"] = len(tnew)
            # Apply the time offset.
            tnew = sbe.sbe56.time_offset(tnew, None, None, drift)

            # Move uncorrected data
            uncorrected = self.data_out.joinpath(uncorrected_name)
            shutil.copy(filename, uncorrected)
            # Remove old file
            filename.unlink()
            # Save corrected data
            tnew.to_netcdf(filename, mode="w")

            # Open and close the newly saved time series to make sure that the old one is not in memory anymore
            tmp = xr.open_dataarray(
                B.get_file_name(455, dir=B.data_out, extension="nc")
            )
            tmp.close()

            # Plot the corrected time series
            cals = self.get_cal_time(455)
            sbe.sbe56.plot(tnew, figure_out=self.figure_out, cal_time=cals)

        # Add a comment to `proc_info`.
        comment = "time vector adjusted based on clock calibrations"
        self.add_comment([sn], comment)
        self.update_proc_info()

    def locate_gaps(self):
        # Find all gaps.
        tdi_all = {ti.attrs["sn"]: find_gaps(ti) for ti in self.allnc}
        gaps = {}
        for sn, tdi in tdi_all.items():
            if len(tdi) > 0:
                gaps[sn] = tdi
                # Subtract gap from time stamp to get time stamp at beginning of gap.
                gaps[sn]["time"] = (tdi.time - tdi).data
        # Find long gaps.
        long_gaps_all = {
            sn: find_first_long_gap(tdi) for sn, tdi in gaps.items()
        }
        long_gaps = {}
        for sn, gi in long_gaps_all.items():
            if ~np.isnat(gi):
                long_gaps[sn] = gi
        # Print date of first long gap.
        print("date of first long gap:")
        _ = [
            print(sn, ":", np.datetime_as_string(gi.time.data, "D"))
            if ~np.isnat(gi)
            else print(sn, ": no long gap")
            for sn, gi in long_gaps_all.items()
        ]
        # Count number of shorter gaps before long gap.
        gaps_up_to_first_long = {}
        n_short_gaps = {}
        for sn, gap in gaps.items():
            if sn not in long_gaps:
                gaps_up_to_first_long[sn] = gap
                n_short_gaps[sn] = len(gap)
            else:
                gaps_up_to_first_long[sn] = gap.where(
                    gap.time <= long_gaps[sn].time, drop=True
                )
                n_short_gaps[sn] = len(gaps_up_to_first_long[sn])
        # Print number of short gaps if any.
        print("number of short gaps:")
        _ = [
            print(sn, ":", ni) if ni > 0 else 0
            for sn, ni in n_short_gaps.items()
        ]
        # Save to class instance
        self.short_gaps = n_short_gaps
        self.first_long_gap = long_gaps

    def comment_no_time_verification_at_end(self):
        no_time_check = [392, 425, 6418, 6432]
        for sni in no_time_check:
            snlist = [int(sni)]
            comment = f"no final warm water clock cal"
            self.add_comment(snlist, comment)

    def comment_no_data(self):
        sn_no_data = self.proc_info[
            ~self.proc_info.raw_data_exists
        ].index.to_numpy()
        for sni in sn_no_data:
            snlist = [int(sni)]
            comment = f"no data"
            self.add_comment(snlist, comment)

    def show_instrument_issues(self):
        """Show instruments that are not ok"""
        tmp = self.proc_info[self.proc_info.comment != "ok"].drop(
            ["utc", "inst", "cal1", "cal2"], axis=1
        )
        display(tmp)

    def find_time_series_terminating_early(self):
        last_times = np.array(
            [thi.time.isel(time=-1).data for thi in self.allnc]
        )
        ind_bool = last_times < np.datetime64("2021-10-04 00:00:00")
        # Extract short time series
        short = [d for d, s in zip(self.allnc, ind_bool) if s]
        sn = [ti.attrs["SN"] for ti in short]
        time = [ti.time[-1].data for ti in short]
        sn = [int(sni[3:]) for sni in sn]
        [print(sni, ":", ti) for sni, ti in zip(sn, time)]
        return sn, time

    def comment_terminates_early(self):
        sn, time = self.find_time_series_terminating_early()
        for sni, ti in zip(sn, time):
            comment = f"terminates {np.datetime_as_string(ti, 'D')}"
            self.add_comment([sni], comment)

    def comment_short_gaps(self):
        for sni, n in self.short_gaps.items():
            if n > 0:
                snlist = [int(sni)]
                if n > 1:
                    comment = f"has {n} short gaps (<1h)"
                else:
                    comment = f"has 1 short gap (<1h)"
                self.add_comment(snlist, comment)

    def comment_long_gap_termination(self):
        for sni, ti in self.first_long_gap.items():
            tstr = np.datetime_as_string(ti.time.data, unit="D")
            snlist = [int(sni)]
            comment = f"terminates {tstr}"
            self.add_comment(snlist, comment)

    def comment_remove_nan_notes(self):
        for sni, note in self.proc_info.Notes.groupby("SN"):
            if type(note.values[0]) == float:
                self.proc_info.Notes.at[sni] = ""

    def comments_to_latex_table(self):
        self.proc_info.to_latex(
            self.doc_dir.joinpath("sbe56_table.tex"),
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

    def load_ctd_cal(self):
        self.cal = xr.open_dataarray(self.ctd_cal)

    def load_mooring_sensor_info(self):
        self.mavs1, self.mavs2 = blt1_load_mooring_sensor_info(
            self.thermistor_info_path, thermistor_type="sbe"
        )

    def calibrate(self, sn):
        xr.set_options(keep_attrs=True)
        th = self.load_nc(sn)
        try:
            sensorcal = self.cal.sel(sn=sn)
            th = th + sensorcal.data
            th.attrs["ctd cal"] = f"ctd cal offset {sensorcal.data:0.3e} added"
        except:
            print(f"no cal for {sn}")
            th.attrs["ctd cal"] = f"no ctd cal available"
        return th

    def cut_and_cal(self, sn):
        # Load thermistor data and apply CTD calibration offset
        th = self.calibrate(sn)
        if sn in self.mavs1.index:
            cut_beg, cut_end = mavs_cut_times(mavsid=1)
        elif sn in self.mavs2.index:
            cut_beg, cut_end = mavs_cut_times(mavsid=2)
        # See if we need to cut short because of a gap > 1h.
        if sn in self.first_long_gap:
            cut_end = self.first_long_gap[sn].time.data
        out = th.sel(time=slice(cut_beg, cut_end), drop=True)
        out.attrs["sn"] = int(out.attrs["SN"][3:])
        return out

    def generate_level1_file(self, sn):
        th = self.cut_and_cal(sn)
        savename = self.level1_dir.joinpath(f"blt1_sbe56_{sn:04}.nc")
        th.to_netcdf(savename)

    def generate_all_level1(self):
        allsn = [ti.attrs["sn"] for ti in self.allnc]
        [self.generate_level1_file(sni) for sni in allsn]

    def load_level1(self, sn):
        return xr.open_dataarray(
            self.level1_dir.joinpath(f"blt1_sbe56_{sn:04}.nc")
        )

    def plot_level1(self, sn):
        th = self.load_level1(sn)
        sbe.sbe56.plot(th)

    def plot_all_level1(self):
        allsn = [ti.attrs["sn"] for ti in self.allnc]
        for sni in tqdm(allsn):
            th = self.load_level1(sni)
            sbe.sbe56.plot(th)
            plotname = self.figure_out_level1.joinpath(
                f"blt1_sbe56_{sni:04}_level1"
            )
            gv.plot.png(plotname, verbose=False)
            plt.close()
            th.close()


# not class
def sbe_get_file_name(sn, data_raw, extension="csv"):
    files = list(data_raw.glob(f"SBE056{sn:05}*.{extension}"))
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        raise OSError(f"more than one file for SN{sn} in {data_raw}")


def sbe_generate_proc_info():
    cal_file = "blt1_sbe56_time_offsets.csv"
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
    proc_info.utc = pd.to_datetime(proc_info.utc, errors="coerce", utc=True)
    proc_info.inst = pd.to_datetime(proc_info.inst, errors="coerce", utc=True)
    proc_info.cal1 = pd.to_datetime(proc_info.cal1, errors="coerce", utc=True)
    proc_info.cal2 = pd.to_datetime(proc_info.cal2, errors="coerce", utc=True)

    n = proc_info.index.shape[0]

    proc_info = proc_info.assign(processed=np.tile(False, n))
    proc_info = proc_info.assign(raw_data_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(figure_exists=proc_info.processed.copy())
    proc_info = proc_info.assign(comment=np.tile("ok", n))

    proc_info = proc_info.sort_index()
    return proc_info


def sbe_update_proc_info(proc_info, data_raw):
    for g, v in proc_info.groupby("SN"):
        try:
            f = get_file_name(g, data_raw=data_raw, extension="csv")
            if f is not None:
                proc_info.raw_data_exists.at[g] = True
            f = get_file_name(g, data_raw=data_out, extension="nc")
            if f is not None:
                proc_info.processed.at[g] = True
            f = get_file_name(g, data_raw=figure_out, extension="png")
            if f is not None:
                proc_info.figure_exists.at[g] = True
        except:
            pass
    return proc_info


def sbe_add_comment(sn, comment, proc_info):
    if type(sn) is float:
        sn = [sn]
    for si in sn:
        if proc_info.comment.at[si] == "ok":
            proc_info.comment.at[si] = comment
        if comment not in proc_info.comment.at[si]:
            addstr = "\\\\"
            proc_info.comment.at[si] = (
                proc_info.comment.at[si] + addstr + comment
            )
    return proc_info


def sbe_show_info():
    """Show instruments that are not ok"""
    tmp = proc_info[proc_info.comment != "ok"].drop(
        ["utc", "inst", "cal1", "cal2"], axis=1
    )
    from IPython.display import display

    display(tmp)
