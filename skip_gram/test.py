import numpy as np
a = np.array([1,2,2,2,2,4,5])
b = np.bincount(a)
b = np.append(b,2)
print(b)
