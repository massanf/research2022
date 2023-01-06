import cupy as np
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
        zm=0.5,
        height=500,
        width=500,
        zoffset=-50,
        pad=0,
        delx=6e-3,
        sdr=500,
        theta=0,
        phi=0,
        gamma=0,
        # cropstartx=100,
        # cropstarty=150,
        # cropheight=400,
        # cropwidth=400
        cropstartx=75,
        cropstarty=120,
        cropheight=350,
        cropwidth=350
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
        self.bx, self.by, self.bz = (np.array(volume.shape) *
                                     np.array(spacing) / 2)

        self.img = np.empty(self.num_views, dtype=object)

        for view in tqdm(range(0, self.num_views), desc="DRR", leave=False):
            a = 2 * np.pi * view / self.num_views
            detector_kwargs = {
                "sdr": sdr,
                "theta": theta,
                "phi": a + phi,
                "gamma": gamma,
                "bx": self.bx,
                "by": self.by + zoffset,
                "bz": self.bz,
            }

            # define the model
            drr = DRR(
                volume,
                spacing,
                width=width,
                height=height,
                delx=delx,
                device="cuda"
            )

            # Make the DRR image
            img_tensor = drr(**detector_kwargs)
            img = img_tensor.to('cpu').detach().numpy().copy()
            img /= np.max(img)
            img *= 255

            self.img[view] = (img.astype("uint8")
                              [cropstarty:cropstarty + cropheight,
                               cropstartx:cropstartx + cropwidth])

            del img
            del img_tensor
            del drr
            torch.cuda.empty_cache()
