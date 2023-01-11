import numpy as np
import cupy as cp
import pathlib
import imageio.v2 as imageio
from tqdm import tqdm_notebook as tqdm
# import imageio.v2 as imageio
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
        sample="xray/colorsample.gif"
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
        self.raw_data = np.empty(sheets, dtype=object)
        for i in tqdm(range(0, sheets)):
            self.load(i, sample)
        self.img = self.raw_data
        print(self.output_height, self.output_width)
        for i in range(0, len(self.img)):
            self.img[i] = cv2.resize(self.img[i].astype("uint8"),
                                     (self.output_height, self.output_width))
            self.img[i] = self.img[i].astype("uint8")

        # cut out black
        imgs = np.stack(self.img)
        oldshape = np.shape(imgs)
        imgs = np.ravel(imgs)
        for idx, px in enumerate(tqdm(imgs)):
            if px < 40:
                imgs[idx] -= px
            elif px < 60:
                imgs[idx] -= - 2 * (px - 40) + 40
        imgs = np.reshape(imgs, oldshape)

        # add border
        # imgs = [cv2.copyMakeBorder(img, 60, 60, 60, 60,
                                #    cv2.BORDER_CONSTANT,
                                #    value=0) for img in imgs]
        self.img = [cv2.resize(cp.asnumpy(img),
                               (height, height)) for img in imgs]

        # histogram match
        self.img_np = []
        template = np.array(Image.open(sample).convert('L')).astype('float32')
        for idx, img in enumerate(self.img):
            self.img_np.append(self.hist_match(img, template))
        for idx, img in enumerate(self.img_np):
            self.img[idx] = np.array(img)

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
            img = np.fromfile(
                f,
                dtype=np.uint16
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
        print(source)
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
