import numpy as np
import os
import os.path as osp
import argparse
import matplotlib.pyplot as plt
Config ={}
# you should replace it with your own root_path
Config['root_path'] = r'a4/polyvore_outfits'
Config['meta_file'] = r'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.005
Config['num_workers'] = 5


def plot_training_curve(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    xrange = np.linspace(1,len(loss)+1,len(loss))
    ax1 = plt.subplot(1,2,1)
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(xrange,loss,color='red',label='loss')
    plt.plot(xrange,val_loss,color='blue',label='val loss')
    plt.legend()

    ax2 = plt.subplot(1,2,2)
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(xrange,acc,color='green',label='accuracy')
    plt.plot(xrange,val_acc,color='yellow',label='val accuracy')
    plt.legend()

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    plt.savefig('curve.png')
    plt.show()
