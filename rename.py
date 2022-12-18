import os
import glob

folder = 'data/sample/ct/'

mn = 100000000000
for file in list(glob.glob(folder + '*.dcm')):
    # print(file)
    # os.rename(file, file[:5] + ".img")
    # reader = open(file)
    # print(folder + file[-8:])
    # os.rename(file, folder + "0" + file[-7:]+'.dcm')
    # newname = folder + "0" + file[-7:]
    num = int(file[-10:-6]) - 1
    # mn = min(mn, num)
    # print(num)
    newname = folder + f"{num:04d}" + ".dcm"
    # print(newname)
    os.rename(file, newname)
    # os.rename(file, file + '.img')
# print(mn)
