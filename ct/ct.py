import pathlib
import pydicom
import cupy as cp
import numpy as np
import glob
from scipy.ndimage import zoom

# filename = 'ct_no_tumor_phantom_raw/001/1-001-0%s.img' % f'{i:03}'
# filename = script_dir / Path('../train_dcm') / Path(volname) /
#  Path(edition) / Path('%s.dcm' % f'{i:03}')
here = pathlib.Path(__file__).parent.parent.resolve()


class ctset():
    def __init__(self, name: str, type):
        self.name = name
        self.zrange = [5000, -5000]
            # self.sheets = len(glob.glob(str(here / "data" / f"{self.name}"
        # / "ct" / "*.dcm")))
        # for i in notebook_tqdm(range(0, self.sheets)):
        # self.raw_data[i] = self.load(i)
        self.raw_data = []

        for idx, file in enumerate(glob.glob(str(here / "data"
                                   / f"{self.name}" / "ct" / "*.dcm"))):
            self.raw_data.append(self.load(file))
        # for idx, datum in enumerate(self.raw_data):
        #     if not hasattr(datum, "ImagePositionPatient"):
        #         print(idx)
        #         datum = self.raw_data[0]

        self.raw_data = ([datum for datum in self.raw_data
                          if hasattr(datum, "ImagePositionPatient")])

        # get position (exclude first and last)
        z_list = []
        for datum in self.raw_data:
            z_list.append(datum.ImagePositionPatient[2])
        z_list = sorted(z_list)
        z_list = cp.unique(z_list)
        self.zrange[0] = z_list[1]
        self.zrange[1] = z_list[len(z_list) - 2]

        self.sheets = len(self.raw_data)

        # self.raw_data = cp.empty(self.sheets, dtype=object)
        self.img = []

        self.raw_data = sorted(self.raw_data,
                               key=lambda x: x.ImagePositionPatient[2])
        self.raw_data = np.flip(self.raw_data)

        if type == "uint8":
            for i in range(0, self.sheets):
                self.img[i] = self.raw_data[i].pixel_array.astype('uint8')
                self.img[i] = 255 - self.img[i]

        if type == "float32":
            for i in range(0, self.sheets):
                a = cp.array(self.raw_data[i].pixel_array.astype('float32'))
                self.img.append(a)
                self.img[i] /= 4095.
                self.img[i] *= 255.
                self.img[i] = self.img[i].astype('uint8')

    def load(self, file: str):
        # file = here / "data" / f"{self.name}" / "ct" / f"{num:04d}.dcm"
        # self.raw_data[num] = pydicom.dcmread(file)
        return pydicom.dcmread(file)
        # self.img[num] = self.raw_data[num].pixel_array.astype('uint8')
        # self.img[num] = 255 - self.img[num]

    def get(self, num: int):
        return self.img[num]

    def pos(self, num: int):
        base = (self.zrange[0] + self.zrange[1]) / 2
        return self.raw_data[num].ImagePositionPatient[2] - base

    def get_volume(self, zm=1, pad=0):
        ds = self.raw_data[int(len(self.raw_data) / 2)]

        # get spacing
        delX, delY = ds.PixelSpacing
        delX, delY = float(delX), float(delY)
        Zs = []
        for datum in self.raw_data:
            Zs.append(datum.ImagePositionPatient[2])

        Zs = cp.unique(Zs)
        delZ = Zs[0]

        delZs = Zs
        delZs = cp.diff(delZs)
        delZ = float(delZs[0])

        # define output shape
        nx, ny = ds.pixel_array.shape
        nz = len(Zs) + 2 * pad
        volume = cp.zeros((nx, nz, ny))

        # create volume
        for idx in range(0, pad):
            volume[:, idx, :] = self.img[10]

        c = 0
        last_z = self.raw_data[0].ImagePositionPatient[2]
        volume[:, c, :] = self.img[0]

        for idx in range(0, len(self.raw_data)):
            if last_z != self.raw_data[idx].ImagePositionPatient[2]:
                c += 1
                volume[:, c + pad, :] = self.img[idx]
                last_z = self.raw_data[idx].ImagePositionPatient[2]

        for idx in range(len(Zs) + pad, len(Zs) + 2 * pad):
            volume[:, idx, :] = self.img[len(self.img) - 20]

        spacing = [delX / zm, delZ / zm, delY / zm]

        volume = zoom(cp.asnumpy(volume), (zm, zm, zm),
                      order=0, mode='nearest')
        volume = cp.asarray(volume)
        # volume *= 1.5
        return volume, spacing
