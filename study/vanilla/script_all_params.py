import yaml
import numpy as np
import torch
from study.vanilla import vanilla_rnn
from itertools import product

# Will be setting the random seed before every run
rand_seed = 2019

# The starting configuration is whatever is in the file below
start_file = './study/vanilla/laptop_rnn.yml'

# Define various parameter values to be explored
params = {	'f_hidden': ['', 'leaky_relu', 'tanh'],
			'prefix': ['3vowels_nocv_men']
			}

# We'll print results to file so that we can access later
out_file = './study/vanilla/results_3vowels_nocv_men_scalelr.txt'

# Temp file we'll use do define configurations
temp_file = './study/vanilla/temp.yml'

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


if __name__ == '__main__':
	start_conf = open(start_file, 'r')
	str_cnf = start_conf.read()
	print_out = open(out_file, 'w')

	# Print out starting configuration
	print('Starting configuration: \n', file=print_out)
	with open(start_file, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
		print(yaml.dump(cfg, default_flow_style=False), file=print_out)

	# Iterate over all parameters that are to be changed
	for current_params in list(product_dict(**params)):
		str_temp = str_cnf
		for key, value in current_params.items():
			# Get the starting value from the file 
			ind_param = str_temp.find(key + ': ')
			ind_val = ind_param + len(key + ': ')
			vc = 0
			val = ''
			c = str_temp[ind_val]
			while c != '\n':
				val += c
				vc += 1
				c = str_temp[ind_val + vc]		
			str_temp = str_temp[:ind_val] + str(value) + str_temp[ind_val+vc:] 

		temp_cnf = open(temp_file, 'w')
		temp_cnf.write(str_temp)
		temp_cnf.close()

		# Run the temporary configuration
		torch.manual_seed(rand_seed)
		np.random.seed(rand_seed)
		(acc_train, acc_test) = vanilla_rnn.main(['--config', temp_file])

		print('For parameters ' + str(current_params) + ' final train and test accuracies are ' 
			+ str(np.float16(acc_train)) + ', ' + str(np.float16(acc_test)), file=print_out)

	print_out.close()