import subprocess
import glob
import time
import sys

num = int(sys.argv[1])

unfinished = ["./data/4267500", "./data/4267574", "./data/4267607", "./data/4274633", "./data/4275166", "./data/4275177", "./data/4275181", "./data/4275599", "./data/4275660", "./data/4275668", "./data/4275683", "./data/4275684", "./data/4275685", "./data/4275686", "./data/4275687", "./data/4275688", "./data/4291628", "./data/4291629", "./data/4383083", "./data/4404140", "./data/4404141", "./data/4404142", "./data/4404302"]

# images = sorted(glob.glob("./pix2pix/datasets/ctfbp/*/*"))
# exists = {}
# for image in images:
#     exists[image.split("/")[-1].split("_")[0]] = True

for idx, vol_path in enumerate(unfinished[num * 4 : (num + 1) * 4]):
    vol = vol_path.split("/")[-1]
    if "__" in vol_path:
        print(f"{vol_path}: passed")
        continue
    subprocess.run(
        [
            "/home/u00606/anaconda3/envs/rec/bin/python3.10",
            "prepare.py",
            vol,
            str(idx + 1 + num * 4),
        ]
    )
    time.sleep(1)
