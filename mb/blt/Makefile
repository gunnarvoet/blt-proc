# Makefile for BLT multibeam processing

USER = gunnar
REMOTEDATADIR = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/dy138-em122-posmv-shallow/discovery/2021/09
REMOTEDATADIR2 = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/dy138-em122-posmv-deep/discovery/2021/09
REMOTEDATADIR3 = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/dy138-em122-posmv-deep/discovery/2021/10
PUBLIC = /Volumes/Public/DY132/Bathymetry
LOCALDATADIR = /Volumes/Iceland/blt2/data/mb

.PHONY: help all

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  sync              to sync data from ship server"
	@echo "  procmb            to process raw multibeam data [mbsystem]"
	@echo "  cleanmb           to clean processed mb data [python]"

all: procmb cleanmb
procmb: grd/blt_canyon_50m.nc
procmbzoom: grd/blt_canyon_zoom_50m.nc
cleanmb: final/blt_canyon_mb_qc.nc

#####################
# FETCH REMOTE DATA #
#####################

# sync: $(REMOTEDATADIR)/*
sync: $(REMOTEDATADIR)/* $(REMOTEDATADIR2)/* $(REMOTEDATADIR3)/*
	for file in $^ ; do \
			echo "syncing" $${file}/ ; \
			rsync -avz $${file}/ $(LOCALDATADIR)/; \
	done

#########################
# PROCESS RAW MULTIBEAM #
#########################

grd/blt_canyon_zoom_50m.nc: blt_mooring_region_mbsystem.sh GV_MBsystem.sh
	./blt_mooring_region_mbsystem.sh

grd/blt_canyon_50m.nc: blt_mbsystem.sh GV_MBsystem.sh
	./blt_mbsystem.sh

final/blt_canyon_mb_qc.nc: grd/blt_canyon_50m.nc
	./mb_merge.py
