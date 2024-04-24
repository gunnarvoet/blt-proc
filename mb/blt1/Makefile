# Makefile for BLT multibeam processing

USER = gunnar
REMOTEDATADIR = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/DY132-EM122-posmv-shallow/discovery/2021/06
REMOTEDATADIR2 = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/dy132-em122-posmv-rockall-trough/discovery/2021/06
REMOTEDATADIR3 = /Volumes/current_cruise/Ship_Systems/Data/Acoustics/EM-122/Raw/dy132-em122-posmv-rockall-trough/discovery/2021/07
PUBLIC = /Volumes/Public/DY132/Bathymetry
LOCALDATADIR = /Volumes/Iceland/blt1/data/mb

.PHONY: help all

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  sync              to sync data from ship server"
	@echo "  procmb            to process raw multibeam data [mbsystem]"
	@echo "  cleanmb           to clean processed mb data [python]"

all: sync procmb cleanmb

#####################
# FETCH REMOTE DATA #
#####################

sync: $(REMOTEDATADIR)/* $(REMOTEDATADIR2)/* $(REMOTEDATADIR3)/*
	for file in $^ ; do \
			echo "syncing" $${file}/ ; \
			rsync -avz $${file}/ $(LOCALDATADIR)/; \
	done

#########################
# PROCESS RAW MULTIBEAM #
#########################

procmb: grd/blt_canyon_50m.nc
	./blt_mbsystem.sh

######################
# CLEAN PROCESSED MB #
######################

cleanmb: final/blt_canyon_mb_qc.nc
	python mb_merge.py
