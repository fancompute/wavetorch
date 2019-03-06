import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn.functional import pad

import wavetorch

if __name__ == '__main__':
    args = wavetorch.io.options.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        args.dev = torch.device('cuda')
    else:
        args.dev = torch.device('cpu')

    torch.set_num_threads(args.num_threads)

    ### Print args summary
    for i in vars(args):
        print('%16s = %s' % (i, vars(args)[i]))
    print('\n')

    ### Load data
    N_classes = len(args.vowels)

    x_train, x_test, y_train, y_test = wavetorch.data.load_selected_vowels(
                                            args.vowels,
                                            gender=args.gender, 
                                            sr=args.sr, 
                                            normalize=True, 
                                            train_size=args.train_size, 
                                            test_size=args.test_size
                                        )

    x_train = x_train.to(args.dev)
    x_test  = x_test.to(args.dev)
    y_train = y_train.to(args.dev)
    y_test  = y_test.to(args.dev)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    ### Define model
    px, py = wavetorch.utils.setup_probe_coords(N_classes, args.px, args.py, args.pd, args.Nx, args.Ny, args.pml_N)
    src_x, src_y = wavetorch.utils.setup_src_coords(args.src_x, args.src_y, args.Nx, args.Ny, args.pml_N)

    if args.design_region:
        design_region = torch.zeros(args.Nx, args.Ny, dtype=torch.uint8)
        design_region[src_x+5:np.min(px)-5] = 1 # For now, just hardcode this in
    else:
        design_region = None
    model = wavetorch.wave.WaveCell(args.dt, args.Nx, args.Ny, src_x, src_y, px, py, pml_N=args.pml_N, pml_p=args.pml_p, pml_max=args.pml_max, c0=args.c0, c1=args.c1, binarized=args.binarized, init_rand=args.init_rand, design_region=design_region)
    model.to(args.dev)

    ### Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = wavetorch.wave.train(model, optimizer, criterion, train_dl, test_dl, args.N_epochs, args.batch_size)
    
    ### Print confusion matrix
    cm_test = wavetorch.utils.calc_cm(model, test_dl)
    cm_train = wavetorch.utils.calc_cm(model, train_dl)

    ### Save model and results
    wavetorch.utils.save_model(model, args.name, history, args, cm_train, cm_test)
