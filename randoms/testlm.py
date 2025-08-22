import numpy as np
import randoms

filename = '/scratch/users/eeganr/0_1.lm'
filenameold = '/scratch/groups/cslevin/eeganr/flangeless/annulus.lm'

# x = np.memmap(filenameold, dtype=np.float32)
# print(x)

testfile = '/scratch/users/eeganr/testcrc/output8Singles.dat'

cylF = (-60.3037, 3.824, 0)

cylD = (52.067, -7.648, 0)

bone = (21.17984085,	-50.89045093, 0)

cylA = (-17.06153846, 56.77374005, 0)

middle = (0, 0, 0)


randoms.test_source(testfile, *cylF, 20)

