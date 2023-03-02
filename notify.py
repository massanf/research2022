import glob, time

now = 0
while 1:
    count = len(glob.glob("./pix2pix/datasets/ctfbp/*/*"))
    # print(count)
    print(count, flush=True)
    if int(count / 1000) > now:
        now = int(count / 1000)
        print(count, "notify", flush=True)
    time.sleep(5)
