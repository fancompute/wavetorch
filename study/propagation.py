'''
    No training in this study, just wave propagation videos
'''
import torch
from wavetorch.wave import WaveCell
from wavetorch.data import load_all_vowels

import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--N_epochs', type=int, default=5)
    argparser.add_argument('--Nx', type=int, default=140)
    argparser.add_argument('--Ny', type=int, default=70)
    argparser.add_argument('--dt', type=float, default=0.707)
    argparser.add_argument('--src_x', type=int, default=21)
    argparser.add_argument('--src_y', type=int, default=35)
    argparser.add_argument('--sr', type=int, default=5000)
    args = argparser.parse_args()

    h  = args.dt * 2.01 / 1.0

    directories_str = ("./data/vowels/a",
                       "./data/vowels/e/",
                       "./data/vowels/o/")

    x, _ = load_all_vowels(directories_str, sr=args.sr, normalize=True)

    # Without PML
    model = WaveCell(args.dt, args.Nx, args.Ny, h, args.src_x, args.src_y, pml_max=0, pml_p=4.0, pml_N=20)
    model.animate(x, batch_ind=0, block=True)

    # With PML
    model = WaveCell(args.dt, args.Nx, args.Ny, h, args.src_x, args.src_y, pml_max=3, pml_p=4.0, pml_N=20)
    model.animate(x, batch_ind=0, block=True)
