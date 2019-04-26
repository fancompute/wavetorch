# Study scripts

This directory contains various study scripts and configuration files for interacting with wavetorch. Some of the scripts, particularly the ones in this folder, are self-contained. However, the `linear/` and `nonlinear_speed/` folders contain yaml configuration files and slurm job scripts for the vowel recognition task described in the paper. 

The yaml files are meant to be used in conjunction with the `vowel_train.py` script in this directory, like so:
```
python ./study/vowel_train.py ./study/example.yml
```
The `vowel_summary.py` and `vowel_analyze.py` scripts are intended to help with analysis of models which were previously trained and saved by `vowel_train.py`. The example configuration file in this directory, `example.yml`, is a good starting point for understanding the configuration options for the vowel recognition problem.

For performing simulations of propagating waves in user-defined geometries, take a look at `propagate.py`.
