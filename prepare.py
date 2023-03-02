import importlib
import imageio.v2 as imageio
from patient import patient
import glob
from tqdm.notebook import tqdm
import json
import socket
import requests
import sys

# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ

import os
import numpy as np
import cupy as cp
import cv2
import argparse
import torch
from multiprocessing import Pool

# setting path

sys.path.append("./../")
from patient import patient

use_range = [130, 220]


files = sorted(glob.glob("./data/*"))
vol_names = {}
for idx, file in enumerate(files):
    vol_names[file.split("/")[2]] = idx


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(
        path_A, 1
    )  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(
        path_B, 1
    )  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


def main():
    # for vol_path in tqdm(glob.glob("./data/*")):
    # if "__" in vol_path:
    # continue
    # vol = vol_path.split("/")[-1]
    # print(vol)
    # print(vol)
    pat_name = sys.argv[1]

    if vol_names[pat_name] < 0.7 * len(vol_names):
        sp = "train"
    elif vol_names[pat_name] < 0.9 * len(vol_names):
        sp = "val"
    else:
        sp = "test"

    p = patient.patient(
        name=pat_name,
        do={
            "ct": True,
            "drr": True,
            "posdrr": True,
            "fbp": True,
            "posfbp": True,
            "resize": True,
        },
        skip_done=True,
        prnt=True,
        idx=int(sys.argv[2])
    )

    img_fold_AB = os.path.join("./pix2pix/datasets/ctfbp", sp)

    path_AB = os.path.join(img_fold_AB, f"{pat_name}_{use_range[0]}.jpg")

    for img in tqdm(range(use_range[0], use_range[1]), desc=pat_name, leave=False):
        path_AB = os.path.join(img_fold_AB, f"{pat_name}_{img}.jpg")
        if os.path.exists(path_AB):
            continue
        im_ct = p.ct.img[img]
        # print("getting")
        im_fbp = p.get_equiv_fbp(img)
        # print("connecting")
        im_AB = cp.asnumpy(cp.concatenate([im_ct, im_fbp], 1)).astype("uint8")

        # print(path_AB)
        # print("saving")
        if not cv2.imwrite("./" + path_AB, im_AB):
            print("NO!")
            sys.exit(0)

    # imageio.mimsave(f"pics/drr.gif", p.drr.img)


try:
    main()
except KeyboardInterrupt:
    raise
except Exception:
    raise
