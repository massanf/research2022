import fbp.tompy as fbp
import cupy as cp
import importlib
from xray import xray
import imageio.v2 as imageio
import cv2
import cupy as cp
import numpy as np
import os, sys
import argparse

# from tqdm.notebook import tqdm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--volname', type=str)           # positional argument
parser.add_argument('--voltage', type=int)           # positional argument
parser.add_argument('--num_views', type=int)      # option that takes a value
parser.add_argument('--use_range_lower', type=int, default=0)
parser.add_argument('--use_range_upper', type=int, default=1024)
parser.add_argument('--spacing', type=int, default=8)

args = parser.parse_args()

volname = args.volname
numsheets = args.num_views
voltage = args.voltage

# which fbp to use
use_range = [args.use_range_lower, args.use_range_upper]
spacing = args.spacing

# load xrays
xrayset = xray.xrayset(
    name=volname, sheets=numsheets, voltage=voltage,
    height=1024, output_height=1024
)

fbpheight = 1000

# fbp from xrayset
rec = fbp.fbpset(
    xrayset, height=fbpheight, angle=50, sidepad=200,
    rotate=195, load_all=False
)


def adjust(img, alpha, beta):
    """Adjust contrast and brightness

    Alpha: contrast, Beta: brightness

    Args:
        img (numpy array): input image

    Returns:
        numpy array: output image
    """
    dst = alpha * img + beta
    return cp.clip(dst, 0, 255).astype(cp.uint8)


cropheight = 600
cropwidth = 600
cropstarty = int((fbpheight - cropheight) / 2)
cropstartx = int((fbpheight - cropwidth) / 2)

imgs = [
    rec.get(i).astype("uint8")
    for i in tqdm(
        range(use_range[0], use_range[1], spacing),
        file=sys.stdout, desc="Getting FBP"
    )
]

# imgs = [rec.get(450).astype("uint8")]

img_fold_AB = "./pix2pix/datasets/ctfbp/test"

for idx, img in enumerate(tqdm(imgs, file=sys.stdout, desc="Saving")):
    cropped = img[
        cropstarty: cropstarty + cropheight,
        cropstartx: cropstartx + cropwidth
    ]
    cropped = cp.array(cv2.resize(cp.asnumpy(cropped), (512, 512)))
    cropped = adjust(cropped, 1.0, 100)
    black = cp.full(cp.shape(cropped), 0)
    im_AB = cp.asnumpy(cp.concatenate([black, cropped], 1)).astype("uint8")
    # imageio.imsave("test.png", im_AB)
    path_AB = os.path.join(img_fold_AB, f"{volname}_{idx:02d}.jpg")
    if not cv2.imwrite("./" + path_AB, im_AB):
        print(f"Image save failed. Please double make sure {img_fold_AB} exists.")
        sys.exit(0)
