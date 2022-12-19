import numpy as np
import sys
import torch
import cv2

from pathlib import Path
from drr.DiffDRR.diffdrr.drr import DRR
from tqdm.autonotebook import tqdm as notebook_tqdm
# from skimage.transform import resize

here = Path(__file__).resolve().parent
sys.path.append(str(here) + "/..")
# from ct import ct # noqa: E402


class drrset():
    def __init__(
        self,
        ctset,
        num_views: int,
        zm=0.5,
        height=300,
        width=300,
        zoffset=0,
        pad=0,
        delx=1e-2,
        sdr=100,
        theta=0,
        phi=0,
        gamma=0
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

        volume, spacing = ctset.get_volume(zm, pad=pad)

        # spacing = [0.247937, 1.0, 0.247937]

        # Get parameters for the detector
        self.bx, self.by, self.bz = (np.array(volume.shape) *
                                     np.array(spacing) / 2)

        # define the model
        self.drr = DRR(
            volume,
            spacing,
            width=width,
            height=height,
            delx=delx,s
            device="cuda"
        )

        self.img = np.empty(self.num_views, dtype=object)

        for view in notebook_tqdm(range(0, self.num_views)):
            detector_kwargs = {
                "sdr": sdr,
                "theta": theta,
                "phi": 2 * np.pi * view / self.num_views + phi,
                # "gamma" : -np.pi / 2,
                "gamma": np.pi + gamma,
                "bx": self.bx,
                "by": self.by + zoffset,
                "bz": self.bz,
            }

            # Make the DRR image
            img_tensor = self.drr(**detector_kwargs)
            img = img_tensor.to('cpu').detach().numpy().copy()
            img /= np.max(img)
            img *= 255
            # img = cv2.resize(img, (1024, 1024))

            self.img[view] = img.astype("uint8")

            img = None
            img_tensor = None
            torch.cuda.empty_cache()
