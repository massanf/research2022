import os
import glob

folder = 'data/sample1/ct/'

# mn = 100000000000
c = 0
for file in list(glob.glob(folder + '*.dcm')):
    # print(file)
    # os.rename(file, file[:5] + ".img")
    # reader = open(file)
    # print(folder + file[-8:])
    # os.rename(file, folder + "0" + file[-7:]+'.dcm')
    # newname = folder + "0" + file[-7:]
    # print(file)
    # num = int(file[-12:-4]) - 56519872
    # mn = min(mn, num)
    # print(num)
    newname = folder + f"{c:04d}" + ".dcm"
    c += 1
    print(newname)
    # os.rename(file, newname)
    # os.rename(file, file + '.img')
# print(mn)
