import numpy as np
from PIL import Image

# print numpy array in full
# np.set_printoptions(threshold=sys.maxsize)

# set width and height

w, h = 1024, 1024

for i in range(1, 200):
    if i % 10 != 0:
        continue

    filename = 'x_no_tumor_phantom_raw/001/Img_0%s_060_300_1024x1024.img' % f'{i:03}'

    with open(filename, 'rb') as f: 
        # Seek backwards from end of file by 2 bytes per pixel 
        f.seek(-w*h*2, 2) 
        img = np.fromfile(f, dtype=np.uint16).reshape((h,w))
    # print(max(np.ravel(np.array(img))))
    # Image.fromarray(img).save('result.png')
    Image.fromarray((img>>4).astype(np.uint8)).save('x_no_tumor_phantom_2d_proj/2D_projection_%s.jpg' % (int(i/10) - 1)) 
