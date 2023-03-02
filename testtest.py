import importlib
import imageio.v2 as imageio
from patient import patient
import glob
import cupy as cp

# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ

from tqdm.notebook import tqdm

vol = "4264284"

# print(vol)
p = patient.patient(name=vol, num_views=450)

# p.generate_drr(cont=False)

# imgs = [cp.asnumpy(img) for img in p.drr.img]
# imageio.mimsave("save.gif", [imgs[0]])
imageio.imsave("test.png", p.get_equiv_fbp(100))
