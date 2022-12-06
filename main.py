import cv2
import imageio.v2 as imageio
import numpy as np
from filtered_back_projection.tompy import fbp
from raw_xray import xray

x = xray.patient(
        name="phantom",
        id=1,
        n=450,
        voltage=120
    )

# x.save(0, "test2.png")

# sino = np.zeros((100, np.shape(x.data[0])[0]))
# for row in range(0, 100):
#     sino[row] = x.data[row][900]

# # sino = cv2.resize(sino, dsize=(100, 200), interpolation=cv2.INTER_CUBIC)
# sino = cv2.rotate(sino, cv2.ROTATE_90_CLOCKWISE)

# f = fbp(height=1024, width=1024)
# f.from_img(sino)
# imageio.imsave("testaaa.png", f.reconstruction)
