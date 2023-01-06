import pydicom
from pathlib import Path
# import numpy as np
import cupy as np
from scipy.ndimage import rotate
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--rotate_num', type=int, default=4, help='')
parser.add_argument('--vol', type=int, default=4, help='')
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))

rotate_num = args.rotate_num

for l in range (0, args.vol):
    volname = 'vol%d' % l
    # create edition output folder (diffdrr input)
    (script_dir / Path('../train_dcm') / Path(volname)).mkdir(parents=True, exist_ok=True)

    # set angle
    angle = 10 / rotate_num

    for k in range (0, rotate_num):
        edition = str(k)

        (script_dir / Path('../train_dcm') / Path(volname) / Path(edition)).mkdir(parents=True, exist_ok=True)
        for i in range (1, 401):
            ds = pydicom.dcmread(script_dir / Path('../raw_dcm') / Path(volname) / Path('1-{:03d}.dcm'.format(i)))
            data = ds.pixel_array
            data = data[::2, ::2] # 1024 -> 512
            # data = cv2.resize(data, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

            data = rotate(data, angle=angle * k - 5, cval=32668, reshape=False)

            ds.PixelData = data.tobytes()
            ds.Rows, ds.Columns = data.shape

            # print the image information given in the dataset
            # print('The information of the data set after downsampling: \n')

            ds.save_as(script_dir / Path('../train_dcm') / Path(volname) / Path(edition) / '{:03d}.dcm'.format(i))
