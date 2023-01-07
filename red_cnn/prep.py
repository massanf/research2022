import os
# import argparse
import numpy as np
import cupy as cp
import glob
from tqdm import tqdm

from patient import patient


def save_dataset(
    save_path='./red_cnn/npy_img/',
    norm_range_min=-1024.0,
    norm_range_max=3072.0
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Create path : {}'.format(save_path))

    use_range = [130, 220]
    vols = glob.glob("./data/*")
    for pat in tqdm((vols), desc="Patient"):
        pat_name = pat.split("/")[-1]
        if "__" in pat_name:
            continue
        p = patient.patient(name=pat_name)
        for img in tqdm(range(use_range[0], use_range[1]),
                        desc=pat_name, leave=False):
            for io in ['input', 'target']:
                f_name = '{}_{}_{}.npy'.format(pat_name, img, io)
                if os.path.exists(os.path.join(save_path, f_name)):
                    continue
                if io == 'target':
                    f_px = cp.asnumpy(p.ct.img[img])
                elif io == 'input':
                    f_px = cp.asnumpy(p.get_equiv_fbp(img))

                f = normalize_(
                    f_px,
                    norm_range_min,
                    norm_range_max
                )

                f_name = '{}_{}_{}.npy'.format(pat_name, img, io)
                np.save(os.path.join(save_path, f_name), f)


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

# # if __name__ == "__main__":
# def prep(
#     save_path='./red_cnn/npy_img/',
#     norm_range_min=-1024.0,
#     norm_range_max=3072.0
# ):
#     # parser.add_argument('--data_path', type=str,
#     # default='./AAPM-Mayo-CT-Challenge/')
#     # parser.add_argument('--save_path', type=str, default='./red_cnn/npy_img/')

#     # parser.add_argument('--test_patient', type=str, default='L506')
#     # parser.add_argument('--mm', type=int, default=3)
#     # parser.add_argument('--norm_range_min', type=float, default=-1024.0)
#     # parser.add_argument('--norm_range_max', type=float, default=3072.0)

#     save_dataset(args)
