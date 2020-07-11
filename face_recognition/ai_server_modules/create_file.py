import pickle
import numpy as np

d = 512                          # dimension
nb = 10000                      # database size              
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb = xb / np.linalg.norm(xb)
pickle.dump(xb, open("database.pickle", "wb"))

filename = dict()
for i in range(nb):
    filename[i] = str(i)

pickle.dump(filename, open("filename.pickle", "wb"))