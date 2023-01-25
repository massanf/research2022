import subprocess
import glob
import time

ln = len(glob.glob("./data/*"))
for idx, vol_path in enumerate(glob.glob("./data/*")):
    if "__" in vol_path:
        continue
    vol = vol_path.split("/")[-1]
    subprocess.run(["python", "prepare.py", vol, str(idx)])
    time.sleep(5)
