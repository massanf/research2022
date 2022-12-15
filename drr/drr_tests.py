# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.animation as animation
import argparse
import torch
# from PIL import Image, ImageEnhance
import imageio.v2 as imageio

from pathlib import Path
import sys
currentdir = Path(__file__).resolve().parent
sys.path.append(str(currentdir)+"/..")
from ct import ct

import os

# from diffdrr import DRR, load_example_ct, read_dicom
from DiffDRR.diffdrr.drr import DRR
# from diffdrr.visualization import plot_drr

parser = argparse.ArgumentParser()
parser.add_argument('--rotate_num', type=int, default=4, help='')
parser.add_argument('--vol', type=int, default=4, help='')
parser.add_argument('--num_views', type=int, default=1, help='')
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))

rotate_num = args.rotate_num
rotate_num = 1
num_views = args.num_views
num_views = 200

print("", flush=True)
print("=== create_drr.py ===", flush=True)
count = 0


volname = 'vol0'
edition = '0'

# create output folder
(script_dir / Path('../document_drr') / Path(volname)).mkdir(parents=True, exist_ok=True)

# Read in the volume
# volume, spacing = read_dicom(script_dir / Path('../data') / Path(volname) / Path("ct"))

c = ct.ctset(name="vol0")

volume, spacing = c.get_volume(0.5)

# Get parameters for the detector
bx, by, bz = np.array(volume.shape) * np.array(spacing) / 2

# define the model
drr = DRR(volume, spacing, width=300, height=300,  delx=1.0e-2, device="cuda")

num_views = 1

# for each angle
for i in range(0, num_views):
    # create output image
    detector_kwargs = {
        "sdr": 100.0,
        "theta": 0,
        "phi": 2 * np.pi * i / num_views,
        # "gamma" : -np.pi / 2,
        "gamma": np.pi,
        "bx": bx,
        "by": by - 20,
        "bz": bz,
    }

    # Make the DRR image
    img_tensor = drr(**detector_kwargs)

    img = img_tensor.to('cpu').detach().numpy().copy()

    # save img
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # fig.add_axes(ax)
    # ax = plot_drr(img, ax=ax)

    imageio.imsave(f"{i}.png", img)
    torch.cuda.empty_cache()
    # fig.savefig(script_dir / Path('../document_drr') / Path(volname) / Path(str((i+18)%200) + '.jpg'))
    # plt.close(fig)

    # im = Image.open(script_dir / Path('../document_drr') / Path(volname) / Path(str((i+18)%200) + '.jpg'))

    # enhancer = ImageEnhance.Contrast(im)
    # im = enhancer.enhance(0.5)

    # enhancer2 = ImageEnhance.Brightness(im)
    # im = enhancer2.enhance(1.2)

    # im = im.crop((170, 220, 830, 880))
    # im = im.crop((170, 140, 830, 800))

    # im.save(script_dir / Path('../document_drr') / Path(volname) / Path(str((i + 18) % 200) + '.jpg'), quality=95)

    # print("DRR: ", i, "/", num_views * rotate_num * args.vol)
    print("drr:", count, "/", num_views * rotate_num * args.vol, "(", int((count / (num_views * rotate_num * args.vol)) * 100), "% )", flush=True)
