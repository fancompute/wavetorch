""" Helper script for plotting memory usage from memory profiler

Install memory_profiler:
	conda install memory_profiler

Profile the code:
	mprof run study/vowel_train.py study/example.yml

This will generate a mprofile dat file which you can then plot with this script, e.g.

	python study/plot_mem.py ./<some_mprof_dat_file_0>.dat ./<some_mprof_dat_file_1>.dat ./<some_mprof_dat_file_2>.dat ...
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('files', nargs='+')

args = parser.parse_args()

fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(4,3))

for file in args.files:
	data = np.loadtxt(file, usecols=(1,2), skiprows=1, delimiter=' ')
	mem = data[:,0]
	t = data[:,1]
	t = t-t.min()
	ax.plot(t, mem/1e3)

ax.set_xlabel('Time (sec)')
ax.set_ylabel('Memory (GB)')
ax.grid()
plt.show()
