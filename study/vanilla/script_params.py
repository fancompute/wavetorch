import yaml
import numpy as np
import torch
from study.vanilla import vanilla_rnn

# Will be setting the random seed before every run
rand_seed = 2019

# The starting configuration is whatever is in the file below
start_file = './study/vanilla/laptop_rnn.yml'

# Define various parameter values to be explored
params = {'N_hidden': [10, 100],
			'sr': [100, 10000],
			'batch_size': [3, 90],
			'W_scale': [0.1, 0.5]
			'L2_reg': [0, 0.1],
			'grad_clip': [0.1, 10],
			'f_hidden': ['relu', 'tanh']
			}

# We'll print results to file so that we can access later
out_file = './study/vanilla/script_results.txt'

# Temp file we'll use do define configurations
temp_file = './study/vanilla/temp.yml'

if __name__ == '__main__':
	start_conf = open(start_file, 'r')
	str_cnf = start_conf.read()
	print_out = open(out_file, 'w')

	# Print out starting configuration
	print('Starting configuration: \n', file=print_out)
	with open(start_file, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
		print(yaml.dump(cfg, default_flow_style=False), file=print_out)

	# Run the starting configuration
	torch.manual_seed(rand_seed)
	np.random.seed(rand_seed)
	(acc_train_st, acc_test_st) = vanilla_rnn.main(['--config', start_file])

	print('Starting conf. train accuracy: %1.2f; test accuracy: %1.2f. \n' % (acc_train_st, acc_test_st), file=print_out)

	# Iterate over all parameters that are to be changed
	for key, values in params.items():
		# Get the starting value from the file 
		ind_param = str_cnf.find(key + ': ')
		ind_val = ind_param + len(key + ': ')
		vc = 0
		val = ''
		c = str_cnf[ind_val]
		while c != '\n':
			val += c
			vc += 1
			c = str_cnf[ind_val + vc]		

		# Iterate over all values of each parameter
		acc_train_temp, acc_test_temp = [], []
		for iv, value in enumerate(values):
			str_temp = str_cnf[:ind_val] + str(value) + str_cnf[ind_val+vc:] 

			temp_cnf = open(temp_file, 'w')
			temp_cnf.write(str_temp)
			temp_cnf.close()

			# Run the temporary configuration
			torch.manual_seed(rand_seed)
			np.random.seed(rand_seed)
			(acc_train, acc_test) = vanilla_rnn.main(['--config', temp_file])

			acc_train_temp.append(acc_train)
			acc_test_temp.append(acc_test)

		print('For parameter ' + key + ' with values ' + str(values) + ' accuracies are, respectively:', file=print_out)
		print('Training: ' + str(np.float16(acc_train_temp)), file=print_out)
		print('Testing: ' + str(np.float16(acc_test_temp)) + '\n', file=print_out)

	print_out.close()