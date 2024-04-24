#! /usr/bin/env bash 

east=$1
west=$2
south=$3
north=$4
size=$5
name=$6
grd_dir=$7
set -x # echo on

## use pwd command ##
mydir="$(pwd)"
echo "My working dir: $mydir"

## convert .all files to .mb59
for ii in *.all; do
  echo $ii
  if [ ! -e ${ii%.all}.mb59 ]; then
    mbcopy -F58/59 -I$ii -O${ii%.all}.mb59
  fi
  echo $ii
  if [ ! -e ${ii%.all}.mb59.inf ]; then
    mbdatalist -I${ii%.all}.mb59 -O
  fi
done

## convert .mb58 files to .mb59
for ii in *.mb58; do
  echo $ii
  if [ ! -e ${ii%.mb58}.mb59 ]; then
    mbcopy -F58/59 -I$ii -O${ii%.mb58}.mb59
  fi
  echo $ii
  if [ ! -e ${ii%.mb58}.mb59.inf ]; then
    mbdatalist -I${ii%.mb58}.mb59 -O
  fi
done

# generate list of all .mb59 files
if [ -e file_list ]; then
  rm file_list
fi
touch file_list

for ii in *.mb59; do
  echo "$ii 59" >> file_list
done
#\rm file_list;touch file_list
#foreach ii (*.mb59)
#echo $ii 59 >> file_list
#end

# this part could be customized for the files of interest
# mbgrid F2 is median filter, better for ratty unedited data, 
# but memory intensive, per mbgrid man page

mbgrid -I"file_list" -F2 -Ofoo -S1 -C2 -R$east/$west/$south/$north -N -E$size/0.0/m! -G3 -M -U216000

if [ ! -e "$grd_dir/grd/" ]; then
  mkdir "$grd_dir/grd/"
fi

mv foo.grd "$grd_dir/grd/foo.grd"

#unset echo
set +x # echo off

#echo --------------------------------------
#echo on local machine execute the following to change to bathymetry is negative convention:
#echo --------------------------------------
#echo ncap -O -s "z=-z" $outfile $outfile_z_minus.nc
#echo ' '
#echo 'run merge_bathy.m to create merged data and save as mat file'

