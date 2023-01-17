import subprocess
import glob
import tqdm
import time

for vol_path in glob.glob("./data/*"):
    if "__" in vol_path:
        continue
    vol = vol_path.split("/")[-1]
    print(vol)
    subprocess.run(["python", "prepare.py", vol])
    time.sleep(5)
