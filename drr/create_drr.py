import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import argparse
from PIL import Image

import os
from pathlib import Path

from diffdrr import DRR, load_example_ct, read_dicom
from diffdrr.visualization import plot_drr

parser = argparse.ArgumentParser()
parser.add_argument('--rotate_num', type=int, default=4, help='')
parser.add_argument('--vol', type=int, default=4, help='')
parser.add_argument('--num_views', type=int, default=1, help='')
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))

rotate_num = args.rotate_num
num_views = args.num_views

print("", flush=True)
print("=== create_drr.py ===", flush=True)
count = 0

for l in range(0, args.vol):
    volname = 'vol%d' % l
    for k in range(0, rotate_num):
        edition = str(k)

        # create output folder
        (script_dir / Path('../train_drr') / Path(volname) / Path(edition)).mkdir(parents=True, exist_ok=True)

        # Read in the volume
        volume, spacing = read_dicom(script_dir / Path('../train_dcm') / Path(volname) / Path(edition))

        spacing = [0.247937, 0.5, 0.247937]

        # Get parameters for the detector
        bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2

        # define the model
        drr = DRR(volume, spacing, width=300, height=300,  delx=1.0e-2, device="cuda")

        # for each angle
        for i in range(0, num_views):
            count += 1
            # create output image
            detector_kwargs = {
                "sdr"   : 135.0,
                "theta" : 0,
                "phi"   : 2 * np.pi * i / num_views,
                # "gamma" : -np.pi / 2,
                "gamma" : np.pi,
                "bx"    : bx,
                "by"    : by,
                "bz"    : bz,
            }

            # Make the DRR image
            img = drr(**detector_kwargs)

            # save img
            fig = plt.figure(figsize=(10, 10))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
            ax = plot_drr(img, ax=ax)
            fig.savefig(script_dir / Path('../train_drr') / Path(volname) / Path(edition) / Path(str(i) + '.jpg'))
            plt.close(fig)

            im = Image.open(script_dir / Path('../train_drr') / Path(volname) / Path(edition) / Path(str(i) + '.jpg'))
            im = im.crop((170, 220, 830, 880))
            im.save(script_dir / Path('../train_drr') / Path(volname) / Path(edition) / Path(str(i) + '.jpg'), quality=95)

            # print("DRR: ", i, "/", num_views * rotate_num * args.vol)
            print("drr:", count, "/", num_views * rotate_num * args.vol, "(", int((count / (num_views * rotate_num * args.vol)) * 100), "% )", flush=True)
