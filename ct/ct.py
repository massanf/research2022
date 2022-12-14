import pathlib
import pydicom
import numpy as np
import glob
from tqdm.autonotebook import tqdm as notebook_tqdm

# filename = 'ct_no_tumor_phantom_raw/001/1-001-0%s.img' % f'{i:03}'
# filename = script_dir / Path('../train_dcm') / Path(volname) /
#  Path(edition) / Path('%s.dcm' % f'{i:03}')
here = pathlib.Path(__file__).parent.parent.resolve()


class ctset():
    def __init__(self, name: str):
        self.name = name
        self.sheets = len(glob.glob(str(here / "data" / f"{self.name}"
                                        / "ct" / "*.dcm")))
        self.raw_data = np.empty(self.sheets, dtype=object)
        self.img = np.empty(self.sheets, dtype=object)
        for i in notebook_tqdm(range(0, self.sheets)):
            self.load(i)
        self.raw_data = sorted(self.raw_data,
                               key=lambda x: x.ImagePositionPatient[2])
        for i in range(0, self.sheets):
            self.img[i] = 255 - self.raw_data[i].pixel_array.astype('uint8')

    def load(self, num: int):
        file = here / "data" / f"{self.name}" / "ct" / f"{num:04d}.dcm"
        self.raw_data[num] = pydicom.dcmread(file)
        # self.img[num] = self.raw_data[num].pixel_array.astype('uint8')
        # self.img[num] = 255 - self.img[num]

    def get(self, num: int):
        return self.img[num]


# ds = pydicom.dcmread(filename)
# data = ds.pixel_array

# print(data)
