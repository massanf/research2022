import os
import glob

# folder = 'raw_xray/phantom/001/120/'

for file in list(glob.glob(folder + '*')):
    # os.rename(file, file[:5] + ".img")
    # reader = open(file)
    # print(folder + file[-8:])
    os.rename(file, folder + file[-26:-22]+'.img')
    # os.rename(file, file + '.img')
