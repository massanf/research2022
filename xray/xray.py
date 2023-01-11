import numpy as np
import cupy as cp
import pathlib
import imageio.v2 as imageio
from tqdm import tqdm_notebook as tqdm
import imageio.v3 as imageio
from PIL import Image
import cv2

here = pathlib.Path(__file__).parent.parent.resolve()


class xrayset():
    """Set of Xray images

    example:
        from xray import xray
        x = xray.xrayset(
            name="phantom",
            id=1,
            sheets=450,
            voltage=120
        )
    """
    def __init__(
        self,
        name: str,
        voltage: int,
        sheets: int,
        height=1024,
        width=0,
        output_height=0,
        output_width=0,
        sample="xray/colorsample.jpg"
    ):
        """init

        Args:
            name (str): Volume name (ex: phantom)
            id (int): Volume id (ex: 1)
            voltage (int): Volume voltage (ex: 60, 120)
            sheets (int): Number of sheets (ex: 200, 450)
            height (int, optional): height of Xray images. Defaults to 1024.
            width (int, optional): width of Xray images. Defaults to 1024.
        """
        self.name = name
        self.voltage = voltage
        self.width = width
        self.height = height
        if width == 0:
            self.width = self.height
        else:
            self.width = width

        if output_height == 0:
            self.output_height = self.height
        else:
            self.output_height = output_height

        if output_width == 0:
            self.output_width = self.output_height
        else:
            self.output_width = output_width

        self.sheets = sheets
        # self.output_height = height
        # self.output_width = width
        # self.raw_data = cp.empty(sheets, dtype=object)
        self.raw_data = [None] * sheets
        # self.raw_data = []
        for i in tqdm(range(0, sheets)):
            self.load(i, sample)
        self.img = self.raw_data
        print(self.output_height, self.output_width)
        for i in range(0, len(self.img)):
            self.img[i] = cp.array(
                            cv2.resize(
                                cp.asnumpy(self.img[i]).astype("uint8"),
                                (self.output_height, self.output_width)
                            )
                        )
            self.img[i] = self.img[i].astype("uint8")

        # cut out black
        imgs = cp.stack(self.img)
        oldshape = cp.shape(imgs)
        imgs = cp.ravel(imgs)

        def filter(px):
            if px < 40:
                return -px
            elif px < 60:
                return - 2 * (px - 40) + 40

        for c in tqdm(range(0, 61)):
            imgs[cp.all(imgs < c)] = filter(c)

        # for idx, px in enumerate(tqdm(imgs)):
        #     if px < 40:
        #         imgs[idx] -= px
        #     elif px < 60:
        #         imgs[idx] -= - 2 * (px - 40) + 40
        imgs = cp.reshape(imgs, oldshape)

        # add border
        imgs = [cv2.copyMakeBorder(cp.asnumpy(img),
                60, 60, 60, 60, cv2.BORDER_CONSTANT,
                value=0) for img in imgs]
        self.img = [cp.array(cv2.resize(cp.asnumpy(img),
                             (height, height))) for img in imgs]

        # histogram match
        self.img_cp = []
        # template = cp.array(imageio.imread(sample,
        #                     index=None)).astype('float32')

        loVal = 110
        for idx, img in enumerate(self.img):
            print(img)
            new_img = img.astype("float64")
            new_img = ((new_img - loVal) * 255.0 / (255 - loVal))
            print(new_img)
            new_img[new_img < 0] = 0
            print(new_img)
            new_img[new_img > 255] = 0
            print(img)
            new_img = new_img.astype(np.uint8)
            # print(new_img)
            self.img_cp.append(cp.array(new_img))
            # self.img_cp.append(img)

        for idx, img in enumerate(self.img_cp):
            self.img[idx] = cp.array(img)

    def load(self, num: int, sample):
        """Load data from picture

        Args:
            num (int): Xray number
        """
        file = (here / "data" / f"{self.name}" / "xray" /
                f"{self.voltage}" / f"{num:04d}.img")

        with open(file, 'rb') as f:
            # Seek backwards from end of file by 2 bytes per pixel
            f.seek(-self.width * self.height * 2, 2)
            img = cp.fromfile(
                f,
                dtype=cp.uint16
            ).reshape((self.height, self.width)).astype("float32")

        img = self.filter(img)
        # img = self.hist_match(img, template)
        self.raw_data[num] = img
        # print(num, img, template, self.hist_match(img, template))

    def filter(self, data):
        """Filter for Xray images

        Args:
            data (_type_): Input data

        Returns:
            _type_: Filtered data
        """
        data *= 255
        data /= 50000
        data *= -1
        data += 250
        # data -= 100
        for datum in data:
            datum[datum < 0] = 0
        return data

    def hist_match(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image
        https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

        Arguments:
        -----------
            source: cp.ndarray
                Image to transform; the histogram is computed over the
                flattened array
            template: cp.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: cp.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding
        # indices and counts
        s_values, bin_idx, s_counts = cp.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = cp.unique(template, return_counts=True)

        s_counts[0] = 0
        t_counts[0] = 0

        # take the cumsum of the counts and normalize by the number of
        # pixels to get the empirical cumulative distribution functions
        # for the source and template images (maps pixel value --> quantile)
        s_quantiles = cp.cumsum(s_counts).astype(cp.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = cp.cumsum(t_counts).astype(cp.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = cp.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    def color_transfer(self, source, target):
        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = self.image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = self.image_stats(target)
        # subtract the means from the target image
        (l, a, b) = cv2.split(target)
        l -= lMeanTar
        a -= aMeanTar
        b -= bMeanTar
        # scale by the standard deviations
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        # add in the source mean
        l += lMeanSrc
        a += aMeanSrc
        b += bMeanSrc
        # clip the pixel intensities to [0, 255] if they fall outside
        # this range
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        # merge the channels together and convert back to the RGB color
        # space, being sure to utilize the 8-bit unsigned integer data
        # type
        transfer = cv2.merge([l, a, b])
        transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

        # return the color transferred image
        return transfer

    def image_stats(self, image):
        # compute the mean and standard deviation of each channel
        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)
