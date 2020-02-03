import numpy as np
a = np.random.randint(9,size=(1,5))
b = np.random.randint(9,size=(5,2))
c = np.random.randint(9,size=(2,1))
print(np.dot(np.dot(a,b),c))
