import sys, os
import multiprocessing, subprocess
from itertools import product
import argparse
from dice_vae import run_dice


def experiment_command(dataset_name, val_reg, margin, epochs, lr, batch_size, encoded_size):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='german', type=str)
    parser.add_argument('--validity_reg', default=84, type=int)
    parser.add_argument('--margin', default=0.165, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=3e-2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--encoded_size', default=10, type=int)
    parser.add_argument('--balanced', default=False, type=lambda x: (str(x).lower() == 'true'), help='If true, uses same number of 0 and 1 labels for training')
    
    args = parser.parse_args()
    args.dataset_name = dataset_name
    args.validity_reg = val_reg
    args.margin = margin
    args.epochs = epochs
    args.lr = lr
    args.batch_size = batch_size
    args.encoded_size = encoded_size

    # print(args)
    run_dice(args)


# dataset_name = ['german']
dataset_name = ['adult']
val_reg = [40, 60, 80, 100]
margin = [0.165]
epochs = [25, 50]
lr = [0.1, 5e-2, 1e-2, 1e-3]
if "german" in dataset_name:
    batch_size = [64, 128]
else:
    batch_size = [512, 1024]
encoded_size = [10, 25, 50]


def hyper_param_explore():
    pool = multiprocessing.Pool(3)
    mr = pool.starmap_async(experiment_command, product(dataset_name, val_reg, margin, epochs, lr, batch_size, encoded_size))
    while not mr.ready():
        sys.stdout.flush()
        mr.wait(0.1)
    print("DONE!")


if __name__ == "__main__":
    hyper_param_explore()
