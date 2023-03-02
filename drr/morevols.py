import pydicom
from pathlib import Path

# import numpy as np
import cupy as np
from scipy.ndimage import rotate
import argparse
import os
import cv2

volnum = 2
gainnum = 3
script_dir = os.path.dirname(os.path.abspath(__file__))

# l = []

count = -1
for vol in range(0, volnum):
    originvolname = "vol" + str(vol)
    # for ver in range (0, 4): #angles
    for transformangle in range(0, 2):
        for transformgain in range(0, gainnum + 1):
            count += 1
            newvolname = "vol" + str(count)
            print(newvolname)
            (script_dir / Path("../raw_dcm") / Path(newvolname)).mkdir(
                parents=True, exist_ok=True
            )

            for i in range(1, 401):
                ds = pydicom.dcmread(
                    script_dir
                    / Path("../original_raw_dcm")
                    / Path(originvolname)
                    / Path("1-{:03d}.dcm".format(i))
                )
                data = ds.pixel_array

                # flip laterally
                # if ver > 1:
                #     data = np.flip(data, 0)

                # transformation
                ratio = 0.9 + (transformgain / (gainnum * 5.0))
                if transformangle == 0:
                    data = cv2.resize(
                        data, dsize=(data.shape[1], int(data.shape[0] * ratio))
                    )  # change width
                    newdata = cv2.resize(data, dsize=(1024, 2048))  # new image
                    newdata[newdata > 0] = 32668  # set to white
                    y_offset = int((2048 - data.shape[0]) / 2)
                    newdata[y_offset : y_offset + data.shape[0], 0:1024] = data
                    data = newdata
                    data = data[512:1536, 0:1024]

                else:
                    data = cv2.resize(
                        data, dsize=(int(data.shape[1] * ratio), data.shape[0])
                    )
                    newdata = cv2.resize(data, dsize=(2048, 1024))  # new image
                    newdata[newdata > 0] = 32668  # set to white
                    x_offset = int((2048 - data.shape[1]) / 2)
                    newdata[0:1024, x_offset : x_offset + data.shape[1]] = data
                    data = newdata
                    data = data[0:1024, 512:1536]

                ds.PixelData = data.tobytes()
                ds.Rows, ds.Columns = data.shape

                # flip vertically
                # if ver % 2 == 1:
                #     ds.ImagePositionPatient[2] = 401 - ds.ImagePositionPatient[2]

                ds.save_as(
                    script_dir
                    / Path("../raw_dcm")
                    / Path(newvolname)
                    / Path("1-{:03d}.dcm".format(i))
                )
