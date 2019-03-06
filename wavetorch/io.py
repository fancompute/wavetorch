import argparse

options = argparse.ArgumentParser()
# Training options
options.add_argument('--name', type=str, default=None,
                            help='Name to add to saved model file. If unspecified a date/time stamp is used')
options.add_argument('--N_epochs', type=int, default=5, 
                            help='Number of training epochs')
options.add_argument('--learning_rate', type=float, default=0.001, 
                            help='Optimizer learning rate')
options.add_argument('--batch_size', type=int, default=3, 
                            help='Batch size used during training and testing')
options.add_argument('--train_size', type=int, default=3,
                            help='Size of randomly selected training set')
options.add_argument('--test_size', type=int, default=3,
                            help='Size of randomly selected testing set')
options.add_argument('--num_threads', type=int, default=4,
                            help='Number of threads')
options.add_argument('--use-cuda', action='store_true',
                            help='Use CUDA to perform computations')

# Data options
options.add_argument('--sr', type=int, default=10000,
                            help='Sampling rate to use for vowel data')
options.add_argument('--gender', type=str, default='men',
                            help='Which gender to use for vowel data. Options are: women, men, or both')
options.add_argument('--vowels', type=str, nargs='*', default=['ei', 'iy', 'oa'],
                            help='Which vowel classes to run on')

# Simulation options
options.add_argument('--c0', type=float, default=1.0,
                            help='Background wave speed')
options.add_argument('--c1', type=float, default=0.9,
                            help='Second wave speed value used with --c0 when --binarized')
options.add_argument('--Nx', type=int, default=140,
                            help='Number of grid cells in x-dimension of simulation domain')
options.add_argument('--Ny', type=int, default=140,
                            help='Number of grid cells in y-dimension of simulation domain')
options.add_argument('--dt', type=float, default=0.707,
                            help='Time step (spatial step size is determined automatically)')
options.add_argument('--px', type=int, nargs='*',
                            help='Probe x-coordinates in grid cells')
options.add_argument('--py', type=int, nargs='*',
                            help='Probe y-coordinates in grid cells')
options.add_argument('--pd', type=int, default=30,
                            help='Spacing, in number grid cells, between probe points')
options.add_argument('--src_x', type=int, default=None,
                            help='Source x-coordinate in grid cells')
options.add_argument('--src_y', type=int, default=None,
                            help='Source y-coordinate in grid cells')
options.add_argument('--binarized', action='store_true',
                            help='Binarize the distribution of wave speed between --c0 and --c1')
options.add_argument('--design_region', action='store_true',
                            help='Set design region')
options.add_argument('--init_rand', action='store_true',
                            help='Use a random initialization for c')
options.add_argument('--pml_N', type=int, default=20,
                            help='PML thickness in grid cells')
options.add_argument('--pml_p', type=float, default=4.0,
                            help='PML polynomial order')
options.add_argument('--pml_max', type=float, default=3.0,
                            help='PML max dampening')

# Inference Options
options.add_argument('--cm', action='store_true',
                            help='Plot the confusion matrix over the whole dataset')
options.add_argument('--show', action='store_true',
                            help='Show the model (distribution of wave speed)')
options.add_argument('--hist', action='store_true',
                            help='Plot the training history from the loaded model')
options.add_argument('--fields', action='store_true',
                            help='Plot the field distrubtion for three classes, STFTs, and simulation energy')
options.add_argument('--animate', action='store_true',
                            help='Animate the field for the  classes')
options.add_argument('--save', action='store_true',
                            help='Save figures')
