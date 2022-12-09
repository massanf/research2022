from pathlib import Path
import matplotlib.pyplot as plt

import os

import numpy as np
from pydicom import dcmread

def read_dicom(dcmdir="data/cxr", correct_zero=True):
    """
    Inputs
    -----
    dcmdir : Path or str
        Path to a DICOM directory
    correct_zero : bool
        Make 0 the minimum value the CT

    Returns
    -------
    volume : ndarray
        3D array containing voxels of imaging data
    spacing : list
        X-, Y-, and Z-directional voxel spacings
    """

    dcmfiles = Path(dcmdir).glob("*.dcm")
    dcmfiles = list(dcmfiles)
    dcmfiles.sort()
    ds = dcmread(dcmfiles[0])

    nx, ny = ds.pixel_array.shape
    nz = len(dcmfiles)
    delX, delY = ds.PixelSpacing
    delX, delY = float(delX), float(delY)
    volume = np.zeros((nx, nz, ny))

    delZs = []
    dss = []

    for idx, dcm in enumerate(dcmfiles):
        ds = dcmread(dcm)
        dss.append([idx, ds])

    dss.sort(key=lambda x: x[1].ImagePositionPatient[2])

    i = 0

    for idx, ds in dss:
        delZs.append(ds.ImagePositionPatient[2])
        arr = ds.pixel_array.astype('float32')
        # ds.pixel_array[ds.pixel_array == 32668] = 127
        # ds.pixel_array[ds.pixel_array == 32644] = 127
        # new_array = 127 - (ds.pixel_array & 127)

        arr -= 32668 # adjust
        arr -= 100 # brightness
        arr *= 6
        arr[arr <= 0] = 0 # cut off below 0
        arr[arr >= 255] = 255 # cut off above 255

        arr /= 255

        arr = np.flip(arr, 0)

        volume[:, i, :] = arr
        i += 1

    delZs = np.diff(delZs)
    delZ = float(np.abs(np.unique(delZs)[0]))

    # spacing = [delX, delY, delZ / 2.]
    spacing = [delX, delZ / 2., delY]
    return volume, spacing


def load_example_ct():
    """Load an example chest CT for demonstration purposes."""
    currdir = Path(__file__).resolve().parent
    # dcmdir = currdir / "cxr_original"
    dcmdir = currdir / "cxr4"
    return read_dicom(dcmdir)
