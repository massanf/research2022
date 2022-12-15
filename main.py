import imageio.v2 as imageio
import fbp.tompy as fbp
from xray import xray
from ct import ct
from drr import drr

xrayset = xray.xrayset(
    name="phantom",
    sheets=450,
    voltage=120
)

rec = fbp.reconstruction(
    xrayset,
    height=300,
    adjust_alpha=4.,
    adjust_beta=-120.,
    rotate=211
)

ctset = ct.ctset(name="vol0")

d = drr.drrset(
    ctset,
    num_views=450,
    zm=0.5,
    height=400,
    width=400,
    zoffset=-25,
    sdr=150
)

for i in range(0, 450):
    imageio.imsave(f"drr_tests/drr{i}.png", d.img[i])
