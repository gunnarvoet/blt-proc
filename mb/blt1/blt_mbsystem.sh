#! /usr/bin/env bash

# Process raw multibeam data from BLT cruise(s) DY132, 

# Set processing directory - make sure to have enough disk space
export PROCESS_DIR=/Volumes/Iceland/blt1/proc/mb/

# Set output directory, most likely where this script resides.
export GRD_DIR=/Users/gunnar/Projects/blt/proc/mb/blt1

# link all .all files into processing directory from as many sources as you have
cd $PROCESS_DIR
ln -s /Volumes/Iceland/blt1/data/mb/*.all   $PROCESS_DIR

# link shell scripts into processing directory
ln -s /Users/gunnar/Projects/blt/proc/mb/blt1*.sh $PROCESS_DIR

# all multibeam data - 50m resolution
# ./GV_MBsystem.sh -14 -10 50 55 50 all $GRD_DIR

# canyon only - 50m resolution
./GV_MBsystem.sh -12.3 -11.75 54.1 54.4 50 all $GRD_DIR

# rename
mv $GRD_DIR/grd/all_50_-12.3_-11.75_54.1_54.4.nc $GRD_DIR/grd/blt_canyon_50m.nc
