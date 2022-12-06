import numpy as np
import pathlib
from tqdm.autonotebook import tqdm as notebook_tqdm
# import imageio.v2 as imageio
# from PIL import Image

here = pathlib.Path(__file__).parent.resolve()


class xrayset():
    def __init__(self, name, id, voltage, sheets, w=1024, h=1024):
        self.name = name
        self.id = id
        self.voltage = voltage
        self.weight = w
        self.height = h
        self.sheets = sheets
        self.raw_data = np.empty(sheets, dtype=object)
        for i in notebook_tqdm(range(0, sheets)):
            self.load(i)
        self.img = self.filter(self.raw_data)
        for i in range(0, len(self.img)):
            self.img[i] = self.img[i].astype("uint8")

    def load(self, num):
        file = (here / f"{self.name}" / f"{self.id:03d}"
                / f"{self.voltage}" / f"{num:04d}.img")

        with open(file, 'rb') as f:
            # Seek backwards from end of file by 2 bytes per pixel
            f.seek(-self.weight * self.height * 2, 2)
            img = np.fromfile(
                f,
                dtype=np.uint16
            ).reshape((self.height, self.weight)).astype("float32")
        self.raw_data[num] = img

    def filter(self, data):
        data *= 255
        data /= 50000
        data *= -1
        data += 250
        # data -= 100
        for datum in data:
            datum[datum < 0] = 0
        return data
