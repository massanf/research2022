import subprocess
import glob
import time

ln = len(glob.glob("./data/*"))
for idx, vol_path in enumerate(glob.glob("./data/*")):
    vol = vol_path.split("/")[-1]
    if "__" in vol_path:
        print(f"{vol_path}: passed")
        continue
    subprocess.run(["python", "prepare.py", vol, str(idx)])
    time.sleep(1)
