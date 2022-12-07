import os
import glob

folder = 'ct/data/vol1/'

for file in list(glob.glob(folder + '*')):
    print(file)
    # os.rename(file, file[:5] + ".img")
    # reader = open(file)
    # print(folder + file[-8:])
    # os.rename(file, folder + "0" + file[-7:]+'.dcm')
    newname = folder + "0" + file[-7:]
    num = int(newname[-8:-4]) - 1
    newname = folder + f"{num:04d}" + ".dcm"
    # print(newname)
    # os.rename(file, newname)
    # os.rename(file, file + '.img')
