import matplotlib.pyplot as plt
from utils import plot_training_curve

# history = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
# history['loss'] = [0.673,0.6588,0.6538,0.6519,0.6487,0.6471,0.6451,0.6443,0.6432]
# history['val_loss'] = [0.6711,0.6574,0.6524,0.6560,0.6485,0.6456,0.6441,0.6457,0.643]
# history['acc'] = [0.5811,0.6065,0.6128,0.6152,0.6183,0.6201,0.6224,0.6236,0.6245]
# history['val_acc'] = [0.585,0.6079,0.6150,0.6070,0.6197,0.6234,0.6238,0.6218,0.6251]
# plot_training_curve(history)

# q2 = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
# q2['loss'] = [2.4111,1.7901,1.6184,1.5147,1.4422,1.4059,1.3296,1.2829,1.2427,1.2047,1.2046,1.1540,1.1162,1.0852,1.0549,1.0267,0.9978,0.9703,0.9444,0.9718]
# q2['val_loss'] = [3.4601,1.8265,1.6757,1.5902,1.8767,1.4622,1.3891,1.3815,1.4252,1.3850,1.3043,1.2228,1.3425,1.3627,1.2572,1.2868,1.2667,1.3292,1.3000,1.371]
# q2['acc'] = [0.3671,0.5029,0.5408,0.5644,0.5814,0.5938,0.6085,0.6200,0.6282,0.6365,0.6405,0.6517,0.6599,0.6675,0.6744,0.6812,0.6887,0.6943,0.7017,0.7089]
# q2['val_acc'] = [0.3253,0.4959,0.5305,0.5523,0.5218,0.5777,0.5964,0.5999,0.5955,0.6047,0.6224,0.6379,0.6112,0.6162,0.6369,0.6280,0.6402,0.6249,0.6379,0.6318]
# plot_training_curve(q2)
"""
loss: 2.4111 - accuracy: 0.3671 - val_loss: 3.4601 - val_accuracy: 0.3273 - lr: 0.0050
loss: 1.7907 - accuracy: 0.5029 - val_loss: 1.8265 - val_accuracy: 0.4959 - lr: 0.0050
loss: 1.6184 - accuracy: 0.5408 - val_loss: 1.6757 - val_accuracy: 0.5305 - lr: 0.0050
loss: 1.5147 - accuracy: 0.5644 - val_loss: 1.5902 - val_accuracy: 0.5523 - lr: 0.0050
loss: 1.4422 - accuracy: 0.5814 - val_loss: 1.8767 - val_accuracy: 0.5218 - lr: 0.0050
loss: 1.4059 - accuracy: 0.5938 - val_loss: 1.4622 - val_accuracy: 0.5777 - lr: 0.0030
loss: 1.3296 - accuracy: 0.6085 - val_loss: 1.3891 - val_accuracy: 0.5964 - lr: 0.0030
loss: 1.2829 - accuracy: 0.6200 - val_loss: 1.3815 - val_accuracy: 0.5999 - lr: 0.0030
loss: 1.2427 - accuracy: 0.6282 - val_loss: 1.4252 - val_accuracy: 0.5955 - lr: 0.0030
loss: 1.2047 - accuracy: 0.6365 - val_loss: 1.3850 - val_accuracy: 0.6047 - lr: 0.0030
loss: 1.2046 - accuracy: 0.6405 - val_loss: 1.3043 - val_accuracy: 0.6224
loss: 1.1540 - accuracy: 0.6517 - val_loss: 1.2228 - val_accuracy: 0.6379
loss: 1.1162 - accuracy: 0.6599 - val_loss: 1.3425 - val_accuracy: 0.6112
loss: 1.0852 - accuracy: 0.6675 - val_loss: 1.3627 - val_accuracy: 0.6162
loss: 1.0549 - accuracy: 0.6744 - val_loss: 1.2572 - val_accuracy: 0.6369
loss: 1.0267 - accuracy: 0.6812 - val_loss: 1.2868 - val_accuracy: 0.6280
loss: 0.9978 - accuracy: 0.6887 - val_loss: 1.2667 - val_accuracy: 0.6402
loss: 0.9703 - accuracy: 0.6943 - val_loss: 1.3292 - val_accuracy: 0.6249
loss: 0.9444 - accuracy: 0.7017 - val_loss: 1.3000 - val_accuracy: 0.6379
loss: 0.9178 - accuracy: 0.7089 - val_loss: 1.3710 - val_accuracy: 0.6318
"""

import numpy as np
import matplotlib.pyplot as plt
acc = [0.6609,0.6902,0.7034,0.7110,0.7092,0.7200,0.7400,0.7421,0.7535,0.7477,0.7475,0.7488,0.7503,0.7531,0.7538]
val_acc = [0.6792,0.6964,0.6988,0.6792,0.7128,0.7210,0.7441,0.7426,0.7454,0.7446,0.7539,0.7545,0.7592,0.7565,0.7556]
xrange = np.linspace(1,len(acc)+1,len(acc))
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(xrange,acc,color='green',label='accuracy')
plt.plot(xrange,val_acc,color='yellow',label='val accuracy')
plt.legend()

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
plt.show()
