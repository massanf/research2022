import numpy as np
import sys
import torch

from pathlib import Path
from drr.DiffDRR.diffdrr.drr import DRR
from tqdm.autonotebook import tqdm as notebook_tqdm

here = Path(__file__).resolve().parent
sys.path.append(str(here) + "/..")
# from ct import ct # noqa: E402


class drrset():
    def __init__(
        self,
        ctset,
        num_views,
        zm=0.5,
        height=300,
        width=300,
        zoffset=0,
        sdr=100
    ):
        self.ctset = ctset
        self.num_views = num_views

        volume, spacing = ctset.get_volume(zm)

        # Get parameters for the detector
        self.bx, self.by, self.bz = (np.array(volume.shape) *
                                     np.array(spacing) / 2)

        # define the model
        self.drr = DRR(
            volume,
            spacing,
            width=width,
            height=height,
            delx=1.0e-2,
            device="cuda"
        )

        self.img = np.empty(self.num_views, dtype=object)

        for view in notebook_tqdm(range(0, self.num_views)):
            detector_kwargs = {
                "sdr": sdr,
                "theta": 0,
                "phi": 2 * np.pi * view / self.num_views,
                # "gamma" : -np.pi / 2,
                "gamma": np.pi,
                "bx": self.bx,
                "by": self.by + zoffset,
                "bz": self.bz,
            }

            # Make the DRR image
            img_tensor = self.drr(**detector_kwargs)
            img = img_tensor.to('cpu').detach().numpy().copy()
            img /= np.max(img)
            img *= 255

            self.img[view] = img.astype("uint8")
            img = None
            img_tensor = None
            torch.cuda.empty_cache()