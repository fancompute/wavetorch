# Study Readme

This folder contains various study configurations. A study is a particular problem configuration described by a YAML file. Some studies also have an associated SLURM job script for running the training routine on a cluster.

The various studies have been organized into sub-folders. However, a good starting point may be `example.yml`, which is in this folder. This study configuration file is the most heavily commented.

A study can be run from the command line (from the top-level directory of the repository) as follows:

```
python -m wavetorch train ./study/example.yml
```

**WARNING:** depending on the batch size, the window length, and the sample rate for the vowel data (all of which are specified in the YAML configuration file) the gradient computation may require a significant amount of memory. It is recommended to start small with the batch size and work your way up gradually, depending on what your machine can handle.
