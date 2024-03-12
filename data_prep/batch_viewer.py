# from GPT-NeoX library with some modifications related to context length

from mmap_dataset import MMapIndexedDataset
from tqdm import trange
import numpy as np
import argparse
import os
import math

from constants import BATCH_SIZE, GRAD_ACCUM_STEPS

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="What train step to start logging"
    )
    parser.add_argument(
        "--end_iteration",
        type=int,
        default=1_000_000 * GRAD_ACCUM_STEPS,
        help="Train step to end logging (inclusive)"
    )
    parser.add_argument(
        "--load_path",
        type = str,
        default = '/mnt/ssd-1/pile_preshuffled/standard/document',
        help = ("MMap dataset path with .bin and .idx files. Omit the .bin (or) .idx "
                "Extension while specifying the path")
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="token_indicies",
        help="Save path for files"
    )
    args = parser.parse_known_args()[0]
    os.makedirs(args.save_path, exist_ok=True)
    filename = os.path.join(args.save_path, "indicies.npy")

    dataset = MMapIndexedDataset(args.load_path, skip_warmup = True)
    # the pre-tokenized sequences are length 2049
    # we want to split these into sequences of length 1024
    indicies = dataset[math.ceil(args.start_iteration/2)*BATCH_SIZE: math.ceil(args.end_iteration/2)*BATCH_SIZE + 1]
    np.save(filename, indicies)
