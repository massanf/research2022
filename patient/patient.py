import pathlib
import numpy as np
import pickle
import cv2
from skopt import gp_minimize
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import fbp.tompy as fbp
from ct import ct
from drr import drr

datadir = pathlib.Path("data")


class patient():
    def __init__(self, name, num_views):
        self.name = name
        self.num_views = num_views
        self.m = -1000
        self.b = -1000
        self.resize_factor = 1

        if (datadir / name / "ct.pickle").exists():
            with open(datadir / name / "ct.pickle", 'rb') as handle:
                self.ct = pickle.load(handle)
        else:
            self.generate_ct()

        if (datadir / name / "drr.pickle").exists():
            with open(datadir / name / "drr.pickle", 'rb') as handle:
                self.drr = pickle.load(handle)
        else:
            self.generate_drr()

        if (datadir / name / "fbp.pickle").exists():
            with open(datadir / name / "fbp.pickle", 'rb') as handle:
                self.fbp = pickle.load(handle)
        else:
            self.generate_fbp()

        if (datadir / name / "resize.pickle").exists():
            with open(datadir / name / "resize.pickle", 'rb') as handle:
                self.resize_factor = pickle.load(handle)
        else:
            self.calculate_resize()

    # generators ----------

    def generate_ct(self):
        print("Creating new CTset")
        self.ct = ct.ctset(name=self.name, type="float32")
        with open(datadir / self.name / "ct.pickle", 'wb') as handle:
            pickle.dump(self.ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.new_drr()

    def generate_drr(self):
        print("Creating new DRRset")
        self.drr = drr.drrset(ctset=self.ct, num_views=self.num_views)
        with open(datadir / self.name / "drr.pickle", 'wb') as handle:
            pickle.dump(self.drr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.new_fbp()

    def generate_fbp(self):
        print("Creating new FBPset")
        self.fbp = fbp.fbpset(
            self.drr,
            height=500,
            angle=75,
            rotate=191
        )
        with open(datadir / self.name / "fbp.pickle", 'wb') as handle:
            pickle.dump(self.fbp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.calculate_resize()

    def calculate_resize(self, base=150, plot=True):
        print("Calculating resize factor")

        def loss(x):
            return self.get_equiv(base, resize_factor=x[0])[1]

        n_calls = 50
        self.result = gp_minimize(
            loss,
            [(1.0, 2.0)],
            n_calls=n_calls,
            callback=[self.tqdm_skopt(total=n_calls)]
        )
        self.resize_factor = self.result.x[0]
        with open(datadir / self.name / "resize.pickle", 'wb') as handle:
            pickle.dump(self.resize_factor, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        if plot:
            plt.scatter(self.result.x_iters, self.result.func_vals, s=1.0)
            plt.title("Resize factor")
            plt.xlabel("resize_factor")
            plt.ylabel("Average pixel value difference")
            plt.show()

    # helpers --------

    def get_equiv(self, num, resize_factor=0, plot=False):
        if resize_factor == 0:
            resize_factor = self.resize_factor
        ttl = np.sum(np.full(np.shape(self.ct.img[0]), 255))
        history = []
        min_sum = 53106966000
        min_sum_idx = 0
        for idx in range(len(self.fbp.x.img[0])):
            now = np.sum(np.absolute(self.ct.img[num] -
                         self.hist_match(self.get_resized_fbp(int(idx),
                                         resize_factor=resize_factor),
                         self.ct.img[num]))) / ttl
            history.append(now)
            if min_sum > now:
                min_sum = now
                min_sum_idx = idx
        # print(min_sum_idx, min_sum, resize_factor)
        # print(min_sum / np.sum(np.full(np.shape(self.ct.img[0]), 255)))
        if plot:
            plt.title("Position matching")
            plt.xlabel("FBP position")
            plt.ylabel("Average pixel value difference")
            plt.plot(history)
        return (min_sum_idx, min_sum)

    def get_equiv_fbp(self, num):
        return self.get_resized_fbp(self.get_equiv(num)[0])

    def get_resized_fbp(self, num, resize_factor=0):
        if resize_factor == 0:
            resize_factor = self.resize_factor
        img = self.fbp.get(num)
        scaled = int(np.shape(img)[0] * resize_factor)
        img = cv2.resize(img, (scaled, scaled))
        out = np.shape(self.ct.img[0])[0]
        new_img = np.zeros(np.shape(self.ct.img[0]))

        if out > scaled:
            start = int((out - scaled) / 2)
            new_img[start:start + scaled, start:start + scaled] = img
        else:
            start = int((scaled - out) / 2)
            # print(start, out)
            new_img = img[start:start + out, start:start + out]
        return new_img

    # utils ---------

    class tqdm_skopt(object):
        def __init__(self, **kwargs):
            self._bar = tqdm(**kwargs)

        def __call__(self, res):
            self._bar.update()

    def hist_match(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image
        https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the
                flattened array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding
        # indices and counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of
        # pixels to get the empirical cumulative distribution functions
        # for the source and template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)