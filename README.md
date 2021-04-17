# movie-watching-control

## Description 
This project contains Python code to help perform intersubject analysis on an 
fMRI movie-watching data. The package can be used interactively in coding 
environments such as Juypter Notebook or run from these scripts.

This project's contents are:

* `local_intersubject_pkg`. This project's primary python package. It contains the following modules:
    * `intersubject.py`: Methods to perform intersubject correlation 
    (ISC) and intersubject functional correlation (ISFC).
    * `nonparametric.py`: Functions to perform nonparametric 
    statistical tests that are used with initial intersubject analysis results.
    * Helper modules `basic_stats.py` and `tools.py`

* Top directory files: 
    * `run_intersubject_analysis.py`: 
        Script to perform intersubject analysis and
        tests automatically, using command line arguments.
    * `script_setup.py`: 
        Script that defines project filepaths and parameters in
        `script_settings.json`.
    * `subjects.py`: 
        File containing subject IDs for this dataset (hardcoded for 
        ease of use).
    * `run_fake_prep.py`: Script to create fake data for testing purposes.

## How to run the analysis

First, run `script_setup.py` to create `script_settings.json`, a configuration file containing useful filepaths and some parameters for the analysis that can also be manually modified afterward. This is mostly useful for running these analyses using scripts.

Note: If you want to run the analysis script without using real Nifti data, then you can also execute `run_fake_prep.py` to create a small, random dataset that can be analyzed instead.   

### Running with scripts
Open any terminal/command line on your computer, then do the following:
```
# Run entire group isc by itself  
> python run_intersubject_analysis.py -e

# Run entire isc with sign-flip permutation test
> python run_intersubject_analysis.py -es

# Run within-between group isc by itself
> python run_intersubject_analysis.py -w

# Run within-between isc with group label permutation test
> python run_intersubject_analysis.py -wl
```


Notes:
- within-between permutation test comparatively much longer to complete since it requires recalculating isc for each iteration.
- currently, entire isc is only valid with sign-flip permutation test; within-between isc is only valid with group label permutation test. 


