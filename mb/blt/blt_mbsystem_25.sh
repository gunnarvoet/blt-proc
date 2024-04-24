#! /usr/bin/env bash

# Process raw multibeam data from BLT cruise(s) DY132, 

# Set processing directory - make sure to have enough disk space
export PROCESS_DIR=/Volumes/Iceland/blt/proc/mb/

# Set output directory, most likely where this script resides.
export GRD_DIR=/Users/gunnar/Projects/blt/proc/mb/blt

# link all .mb59 files into processing directory from as many sources as you have
cd $PROCESS_DIR
ln -s /Volumes/Iceland/blt1/proc/mb/*.mb59   $PROCESS_DIR
ln -s /Volumes/Iceland/blt1/proc/mb/*.mb59.inf  $PROCESS_DIR
ln -s /Volumes/Iceland/blt1/proc/mb/*.mb59.fnv  $PROCESS_DIR
ln -s /Volumes/Iceland/blt1/proc/mb/*.mb59.fbt  $PROCESS_DIR
# ln -s /Volumes/Iceland/blt2/proc/mb/*.mb59*   $PROCESS_DIR

# link shell scripts into processing directory
ln -s /Users/gunnar/Projects/blt/proc/mb/blt/*.sh $PROCESS_DIR

# all multibeam data - 50m resolution
# ./GV_MBsystem.sh -14 -10 50 55 50 all $GRD_DIR

# canyon only - 50m resolution
# ./GV_MBsystem.sh -12.3 -10.0 54.0 56.0 50 all $GRD_DIR
./GV_MBsystem.sh -12.3 -11.75 54.1 54.4 25 all $GRD_DIR

# rename
mv $GRD_DIR/grd/foo.grd $GRD_DIR/grd/blt_canyon_25m.nc
mv $PROCESS_DIR/foo_sd.grd $GRD_DIR/grd/blt_canyon_25m_sd.nc
mv $PROCESS_DIR/foo_num.grd $GRD_DIR/grd/blt_canyon_25m_num.nc
