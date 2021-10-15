BLT Mooring Data Processing
===========================

Python processing files were written in Jupyter notebooks and converted to python scripts using [jupytext](https://jupytext.readthedocs.io/en/latest/). They can be converted back to Jupyter notebooks.

A [conda](https://docs.conda.io/en/latest/) environment with all packages needed for running the python processing scripts can be created by running `conda env create -f environemnt.yml`. The newly created environment will be called `blt-proc`.

The `Makefile` bundles a number of data synchronization and processing steps. Note that you need GNU make version 4.3 or higher for this to work properly. On Macs, this can be installed via `brew install make` using [homebrew](https://brew.sh/). Type `make help` to see various options for running the Makefile.

### ADCP

Some of the processing code lives in a separate repository at https://github.com/gunnarvoet/gadcp. This interfaces heavily with the UH pycurrents module; follow the [instructions](https://currents.soest.hawaii.edu/ocn_data_analysis/installation.html) to install the software.

<!-- The following is a list of ADCPs and for how long they recorded data. -->
     
<!-- |  SN |Mooring|Performance| -->
<!-- |-----|-------|-----------| -->
<!-- | 3109|M1     |Full record| -->
<!-- | 9408|M1     |Full record| -->
<!-- |13481|M1     |Full record; issues with pressure time series| -->
<!-- |14408|M1     |Few days only| -->
<!-- |22476|M1     |Few days only| -->
<!-- | 3110|M2     |Full record| -->
<!-- | 8063|M2     |No data| -->
<!-- | 8065|M2     |Few days only; no pressure| -->
<!-- |10219|M2     |Full record| -->
<!-- |22479|M2     |Few days only| -->
<!-- |23615|M2     |Few days only| -->
<!-- |  344|M3     |No data| -->
<!-- | 8122|M3     |Few days only; no pressure| -->
<!-- |12733|M3     |Few days only| -->
<!-- |15339|M3     |Few days only| -->
<!-- |15694|M3     |Full record| -->


**Magnetic declination** at the mooring sites is about xxx.

### SBE37
Some of the processing code lives in a separate repository at https://github.com/gunnarvoet/sbemoored.

**Issues (solved):** **12710, 12711** and **1209** stop sampling at their specified rate pre-maturely. Most likely due to drained batteries - apparently the little tool Seabird provides for calculating instrument endurance isn't that good and one has to be more conservative.
<!-- | SN  | last good data | -->
<!-- |-----|----------------| -->
<!-- |12710|2020-02-09 21:00| -->
<!-- |12711|2020-03-17 12:00| -->
<!-- |12712|2020-01-16 23:15| -->

The clocks of the affected instruments still seemed fine on recovery, time offsets were scaled linearly to the good part of the time series.

### SBE56
Some of the processing code lives in a separate repository at https://github.com/gunnarvoet/sbemoored.

### RBR Solo
Some of the processing code lives in a separate repository at https://github.com/gunnarvoet/rbrmoored.
