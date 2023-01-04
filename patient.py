import pathlib
import numpy as np
import pickle
import cv2
import math
import random
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

        # if (datadir / name / (name + ".pickle")).exists():
        #     self.load()
        # else:
        #     self.new_ct()

        # self.new_ct()

        if (datadir / name / "ct.pickle").exists():
            with open(datadir / name / "ct.pickle", 'rb') as handle:
                self.ct = pickle.load(handle)
        else:
            self.new_ct()

        if (datadir / name / "drr.pickle").exists():
            with open(datadir / name / "drr.pickle", 'rb') as handle:
                self.drr = pickle.load(handle)
        else:
            self.new_drr()

        if (datadir / name / "fbp.pickle").exists():
            with open(datadir / name / "fbp.pickle", 'rb') as handle:
                self.fbp = pickle.load(handle)
        else:
            self.new_fbp()

    def new_ct(self):
        print("Creating new CTset")
        self.ct = ct.ctset(name=self.name, type="float32")
        with open(datadir / self.name / "ct.pickle", 'wb') as handle:
            pickle.dump(self.ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.new_drr()

    def new_drr(self):
        print("Creating new DRRset")
        self.drr = drr.drrset(ctset=self.ct, num_views=self.num_views)
        with open(datadir / self.name / "drr.pickle", 'wb') as handle:
            pickle.dump(self.drr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.new_fbp()

    def new_fbp(self):
        print("Creating new FBPset")
        self.fbp = fbp.fbpset(
            self.drr,
            height=500,
            angle=75,
            rotate=191
        )
        with open(datadir / self.name / "fbp.pickle", 'wb') as handle:
            pickle.dump(self.fbp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_equiv(self, num, resize_factor=0):
        if resize_factor == 0:
            resize_factor = self.resize_factor
        ttl = np.sum(np.full(np.shape(self.ct.img[0]), 255))
        min_sum = 53106966000
        min_sum_idx = 0
        for idx in range(len(self.fbp.x.img[0])):
            now = np.sum(np.absolute(self.ct.img[num] -
                         self.hist_match(self.get_resized_fbp(int(idx),
                                         resize_factor=resize_factor),
                         self.ct.img[num]))) / ttl
            if min_sum > now:
                min_sum = now
                min_sum_idx = idx
        print(min_sum_idx, min_sum, resize_factor)
        # print(min_sum / np.sum(np.full(np.shape(self.ct.img[0]), 255)))
        return (min_sum_idx, min_sum)

    def get_equiv_fbp(self, num):
        return self.get_resized_fbp(self.get_equiv(num)[0])

    def adjust(self, img, alpha=1.0, beta=0.0):
        dst = alpha * img + beta
        return np.clip(dst, 0, 255).astype(np.uint8)

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

    # def save(self):
    #     if self.m == -1000 or self.b == -1000:
    #         print("Failed to save. Position not set.")
    #     else:
    #         with open(datadir / self.name /
    #                   (self.name + ".pickle"), 'wb') as handle:
    #             pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # def load(self):
    #     with open(datadir / self.name /
    #               (self.name + ".pickle"), 'rb') as handle:
    #         loaded = pickle.load(handle)
    #     self.ct = loaded.ct
    #     self.drr = loaded.drr
    #     self.fbp = loaded.fbp
    #     self.m = loaded.m
    #     self.b = loaded.b
    #     self.name = loaded.name
    #     self.num_views = loaded.num_views
