import os
import argparse
import numpy as np
from tqdm import tqdm

from patient import patient


def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_list = args.patient_list
    use_range = [50, 220]
    for p_ind, pat in tqdm(enumerate(patient_list)):
        p = patient.patient(name=pat)
        for img in tqdm(range(use_range[0], use_range[1])):
            for io in ['input', 'target']:
                if io == 'target':
                    f_px = p.ct.img[img]
                elif io == 'input':
                    f_px = p.get_equiv_fbp(img)

                f = normalize_(
                    f_px,
                    args.norm_range_min,
                    args.norm_range_max
                )

                f_name = '{}_{}_{}.npy'.format(pat, img, io)
                np.save(os.path.join(args.save_path, f_name), f)

        # printProgressBar(p_ind, len(patient_list),
                        #  prefix="save image ..",
                        #  suffix='Complete', length=25)
        # print(' ')


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/
    # 91fa21b9dda9d055a02ecd23f24fbc3d
    percent = (("{0:." + str(decimals) + "f}")
               .format(100 * (iteration / float(total))))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_path', type=str,
    # default='./AAPM-Mayo-CT-Challenge/')
    parser.add_argument('--save_path', type=str, default='./red_cnn/npy_img/')

    # parser.add_argument('--test_patient', type=str, default='L506')
    # parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    args = parser.parse_args()
    args.patient_list = [
        "sample1000",
        "sample1001",
        "sample1002",
        "sample1003",
        "sample1004",
        "sample1005",
        "sample1006",
        "sample1007",
        "sample1008",
        "sample1009",
        "sample1010",
    ]
    save_dataset(args)
