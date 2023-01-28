import cupy as cp
import numpy as np
import sys
import torch
# import gc

from pathlib import Path
from drr.DiffDRR.diffdrr.drr import DRR
# from tqdm import tqdm
from tqdm.notebook import tqdm
# from skimage.transform import resize

here = Path(__file__).resolve().parent
sys.path.append(str(here) + "/..")
# from ct import ct # noqa: E402


class drrset():
    def __init__(
        self,
        ctset,
        num_views: int,
        zm=0.7,
        height=500,
        width=500,
        zoffset=-50,
        pad=0,
        sdr=500,
        theta=0,
        phi=0,
        gamma=0,
        # delx=6e-3,
        # cropstartx=75,
        # cropstarty=120,
        # cropheight=350,
        # cropwidth=350,
        delx=5e-3,
        cropstartx=50,
        cropstarty=80,
        cropheight=400,
        cropwidth=400,
        adjust=True
    ):
        """Generates DRR dataset

        Args:
            ctset (ctset): CT set
            num_views (int): number of views
            zm (float, optional): 間引き value. Defaults to 0.5 (half the data).
            height (int, optional): height of OUTPUT in px. Defaults to 300.
            width (int, optional): width of OUTPUT in px. Defaults to 300.
            zoffset (int, optional): view offset of Z axis. Defaults to 0.
            pad (int, optional): padding thickness on  top and bottom.
                                 Defaults to 0.
            delx (_type_, optional): pic area size. Defaults to 1e-2.
            sdr (int, optional): distance from camera. Defaults to 100.
        """
        self.height = height
        self.width = width
        self.ctset = ctset
        self.num_views = num_views
        self.output_height = cropheight
        self.output_width = cropwidth

        volume, spacing = ctset.get_volume(zm, pad=pad)

        # print(spacing)

        # spacing = [0.247937, 1.0, 0.247937]

        # Get parameters for the detector
        self.bx, self.by, self.bz = (cp.shape(volume) *
                                     np.array(spacing) / 2)

        # self.img = cp.empty(self.num_views, dtype=object)
        self.img = [None] * self.num_views

        # define the model
        drr = DRR(
            volume,
            spacing,
            width=width,
            height=height,
            delx=delx,
            device="cuda"
        )

        for view in tqdm(range(0, self.num_views), desc="DRR", leave=False):
            a = 2 * cp.pi * view / self.num_views
            detector_kwargs = {
                "sdr": sdr,
                "theta": theta,
                "phi": a + phi,
                "gamma": gamma,
                "bx": self.bx,
                "by": self.by + zoffset,
                "bz": self.bz,
            }

            # Make the DRR image
            img_tensor = drr(**detector_kwargs)
            img = cp.array(img_tensor.to('cpu').detach().numpy().copy())
            img /= cp.max(img)
            img *= 255

            self.img[view] = (img.astype("uint8")
                              [cropstarty:cropstarty + cropheight,
                               cropstartx:cropstartx + cropwidth])

            # if adjust:
                # self.img[view] = self.f(self.img[view])

            del img
            del img_tensor
            torch.cuda.empty_cache()

        if adjust:
            s = cp.ravel(cp.stack(self.img))
            b = int(cp.percentile(s[s > 50], 40))
            print(b)
            for idx, img in enumerate(self.img):
                # self.img[idx][img > b] = b
                # b = int(cp.percentile(img[img > 50], 70))
                newimg = self.img[idx]
                for c in range(b, 255):
                    newc = self.filter_diffuse(c, b, b + (255 - b) * 0.2)
                    newimg[newimg == c] = newc
                self.img[idx] = newimg

        for idx, img in enumerate(self.img):
            self.img[idx] = self.f(img)

    def filter_diffuse(self, px, breakpoint, highest):
        if px > breakpoint:
            h = highest
            b = breakpoint
            return int((h - b) / (255 - b) * (px - b) + b)
        else:
            return px

    def filter_black(self, px):
        if px < 40:
            return 0
        elif px < 60:
            return px - 2 * (px - 40) + 40

    def f(self, img):
        cutoff = 10
        newimg = img.astype("float64")
        for c in range(0, 61):
            newimg[newimg == c] = self.filter_black(c)
        newimg[newimg > cutoff] = ((newimg[newimg > cutoff] -
                                    cp.average(newimg[newimg > cutoff])) /
                                   cp.std(newimg[newimg > cutoff]) * 28.069 +
                                   202.102)
        newimg[newimg < 0] = 0
        newimg[newimg > 255] = 255
        newimg = newimg.astype("uint8")
        return newimg
        # return img
