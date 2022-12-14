from __future__ import division
# import scipy
import cv2
import pathlib
import numpy as np
import imageio.v2 as imageio
# import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
# from gui import imagename
from skimage.transform import rotate
from xray import xray

here = pathlib.Path(__file__).parent.resolve()


class reconstruction():
    """Reconstruct FBP from xrayset class

    example:
        import filtered_back_projection.tompy as fbp
        rec = fbp.reconstruction(
            x,
            height=300,
            adjust_alpha=4.,
            adjust_beta=-120.,
            rotate=211
        )
        imageio.imsave("rec.png", rec.get(565))

    """
    def __init__(
        self,
        xray: xray.xrayset,
        height: int,
        width=0,
        adjust_alpha=1.0,
        adjust_beta=0.0,
        start=50,
        angle=30,
        sidepad=0,
        toppad=0,
        rotate=211
    ):
        """init

        Args:
            xray (xray.xrayset): xray.xrayset object
            height (int): height of OUTPUT
            width (int, optional): width of OUTPUT, defaults to height
            start (int, optional): start position of sinogram. Defaults to 50.
            angle (int, optional): angle of each slice. Defaults to 30.
            sidepad (int, optional): padding on sinogram. Defaults to 0.
            toppad (int, optional): padding on sinogram. Defaults to 0.
            adjust_alpha (float, optional): for image post processing.
                                            Defaults to 1.0.
            adjust_beta (float, optional): for image post processing.
                                           Defaults to 0.0.
            rotate (int, optional): for image post processing. Defaults to 211.
        """
        self.x = xray
        self.height = height
        if width == 0:
            self.width = height
        else:
            self.width = width

        self.data = np.empty(self.x.height, dtype=object)
        self.loaded = np.full(self.x.height, False)

        self.start = start
        self.angle = angle
        self.sidepad = sidepad
        self.toppad = toppad

        self.adjust_alpha = adjust_alpha
        self.adjust_beta = adjust_beta

        self.rotate = rotate

    def get(self, sheet_number: int) -> np.array:
        """Get FBP result image.

        If not generated yet, it will generate a new one

        Args:
            sheet_number (int): sheet number

        Returns:
            numpy array: result image
        """
        if not self.loaded[sheet_number]:
            self.generate(sheet_number)
        return self.data[sheet_number]

    def generate(
        self,
        sheet_number: int
    ):
        """Generate FBP result

        Args:
            sheet_number (int): sheet number (0 to 1024)
        """
        sinogram = self.create_sino(sheet_number)
        sinogram = cv2.resize(sinogram, (self.height, self.width))

        imageio.imsave(here / "sino.png", sinogram)

        theta = np.linspace(0., 180., max(self.height, self.width),
                            endpoint=False)
        reconstruction_fbp = self.iradon_transform(
            sinogram,
            theta=theta,
            interpolation='cubic').astype("float32")
        reconstruction_fbp -= np.min(reconstruction_fbp.flatten())
        reconstruction_fbp /= np.max(reconstruction_fbp.flatten())
        reconstruction_fbp *= 256
        reconstruction_fbp = reconstruction_fbp.astype("uint8")

        reconstruction_fbp = self.adjust(reconstruction_fbp)
        rec_img = Image.fromarray(reconstruction_fbp)
        rec_img = rec_img.rotate(self.rotate)

        self.data[sheet_number] = np.array(rec_img)
        self.loaded[sheet_number] = True

    def create_sino(self, sheet_number: int):
        """Create sinogram from Xray

        Args:
            sheet_number (int): sheet number (0 to 1024)

        Returns:
            numpy array: sinogram
        """
        n = int(self.x.sheets / 2)
        w = np.shape(self.x.img[0])[0]
        new_sino = np.zeros(
            (n + 2 * self.toppad, w + 2 * self.sidepad),
            dtype="uint8"
        )
        for row in range(self.start, self.start + n):
            for col in range(0, w):
                a = int((row - self.start) % n)
                b = int(row - int(self.angle * col / w))
                new_sino[a][col] = self.x.img[b][sheet_number][col]
        new_sino = np.rot90(new_sino)
        return new_sino

        # image = imageio.imread(
        #             'filtered_back_projection/shepplogan.png',
        #             as_gray=True
        #         ).astype('float64')
        # TODO zapytanie o obrazek gui
        # image = imageio.imread(imagename, as_gray = True).astype('float64')
        # # TODO zapytanie o obrazek gui
        # image  = 255.0 - image.reshape(784)
        # if len(image.shape) == 3:
        #     image = grey_scale(image)
        # # image = misc.imresize(image,(220,220))
        # image = resize(image, (220, 220))
        # sinogram = radon_transform(image, 220)
        # imageio.imsave('test.jpg', sinogram)

    # def from_path(self, sino_path: pathlib.Path, height, width):
    #     sinogram = imageio.imread(sino_path, as_gray=True)
    #     theta = np.linspace(0., 180., max(height, width),
    #                         endpoint=False)
    #     reconstruction_fbp = self.iradon_transform(sinogram,
    #                                         theta=theta,
    #                                         interpolation='cubic')
    #     return reconstruction_fbp
        # return reconstruction_fbp

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        # ax1.set_title("Original")
        # ax1.imshow(image, cmap=plt.cm.Greys_r)
        # ax2.set_title("Radon transform\n(Sinogram)")
        # ax2.set_xlabel("Projection angle (deg)")
        # ax2.set_ylabel("Projection position (pixels)")
        # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
        #            extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        # fig.tight_layout()
        # plt.savefig("test1.png")

        # self.error = reconstruction_fbp - image
        # print('FBP reconstruction error: %.3g'
        #   % np.sqrt(np.mean(self.error**2)))
        # imageio.imsave('asdf.png', reconstruction_fbp)
        # imkwargs = dict(vmin=-0.2, vmax=0.2)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
        #                                sharex=True, sharey=True,
        #                                subplot_kw={'adjustable': 'box'})
        # ax1.set_title("Reconstruction\nFiltered back projection")
        # ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        # ax2.set_title("Reconstruction error\nFiltered back projection")
        # ax2.imshow(reconstruction_fbp-image,cmap=plt.cm.Greys_r, **imkwargs)
        # plt.show()
        # plt.savefig("test2.png")

    def grey_scale(self, photo):
        a = photo[:, :, 0] * 0.299
        b = photo[:, :, 1] * 0.587
        c = photo[:, :, 2] * 0.114
        return a + b + c

    def radon_transform(self, image, steps):
        """Create sinogram from image array
        """
        radon = np.zeros((steps, len(image)), dtype='float64')
        for s in range(steps):
            rotation = rotate(image, -s*180/steps).astype('float64')
            radon[:, s] = sum(rotation)
        return radon

    def sinogram_circle_to_square(self, sinogram):
        diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
        pad = diagonal - sinogram.shape[0]
        old_center = sinogram.shape[0] // 2
        new_center = diagonal // 2
        pad_before = new_center - old_center
        pad_width = ((pad_before, pad - pad_before), (0, 0))
        return np.pad(sinogram, pad_width, mode='constant', constant_values=0)

    def iradon_transform(
        self,
        radon_image,
        theta=None,
        interpolation='linear'
    ):
        """FBP

        Args:
            radon_image (_type_): sinogram
            theta (_type_, optional): image size. Defaults to None.
            interpolation (str, optional): No idea. Defaults to 'linear'.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        output_size = radon_image.shape[0]
        radon_image = self.sinogram_circle_to_square(radon_image)
        th = (np.pi / 180.0) * theta
        # resize image to next power of two (but no less than 64) for
        # Fourier analysis; speeds up Fourier and lessens artifacts
        projection_size_padded = \
            max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
        pad_width = (
                        (0, projection_size_padded - radon_image.shape[0]),
                        (0, 0)
                    )
        img = np.pad(
                        radon_image,
                        pad_width,
                        mode='constant',
                        constant_values=0
                    )
        f = fftfreq(projection_size_padded).reshape(-1, 1)  # digital frequency
        # omega = 2 * np.pi * f                             # angular frequency
        fourier_filter = 2 * np.abs(f)                      # ramp filter
        projection = fft(img, axis=0) * fourier_filter
        radon_filtered = np.real(ifft(projection, axis=0))
        radon_filtered = radon_filtered[:radon_image.shape[0], :]
        reconstructed = np.zeros((output_size, output_size))
        mid_index = radon_image.shape[0] // 2
        [X, Y] = np.mgrid[0:output_size, 0:output_size]
        xpr = X - int(output_size) // 2
        ypr = Y - int(output_size) // 2
        # Reconstruct image by interpolation
        interpolation_types = ('linear', 'nearest', 'cubic')
        if interpolation not in interpolation_types:
            raise ValueError("Unknown interpolation: %s" % interpolation)
        for i in range(len(theta)):
            t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
            x = np.arange(radon_filtered.shape[0]) - mid_index
            if interpolation == 'linear':
                backprojected = np.interp(t, x, radon_filtered[:, i],
                                          left=0, right=0)
            else:
                interpolant = interp1d(x, radon_filtered[:, i],
                                       kind=interpolation,
                                       bounds_error=False, fill_value=0)
                backprojected = interpolant(t)
            reconstructed += backprojected
        radius = output_size // 2
        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
        reconstructed[~reconstruction_circle] = 0.

        return reconstructed * np.pi / (2 * len(th))

    def adjust(self, img):
        """Adjust contrast and brightness

        Alpha: contrast, Beta: brightness

        Args:
            img (numpy array): input image

        Returns:
            numpy array: output image
        """
        dst = self.adjust_alpha * img + self.adjust_beta
        return np.clip(dst, 0, 255).astype(np.uint8)
