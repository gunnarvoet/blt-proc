BLT Mooring Data Processing
===========================

Python processing files were written in Jupyter notebooks and converted to python scripts using [jupytext](https://jupytext.readthedocs.io/en/latest/). They can be converted back to Jupyter notebooks.

A [conda](https://docs.conda.io/en/latest/) environment with all packages needed for running the python processing scripts can be created by running `conda env create -f environemnt.yml`. The newly created environment will be called `blt-proc`.

The `Makefile` bundles a number of data synchronization and processing steps. Note that you need GNU `make` version 4.3 or higher for this to work properly. Under macOS, this can be installed via `brew install make` using [homebrew](https://brew.sh/). Type `make help` to see various options for running the Makefile.

See [doc](https://github.com/gunnarvoet/blt-proc/blob/main/doc) for processing notes.
