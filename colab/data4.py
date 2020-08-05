from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

import tensorflow
from utils import Config

class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        # map id to img pairs 2 compatibility
        outfits = open(self.root_dir+'/train.json', 'r')
        outfits = json.load(outfits)
        set = {}
        for outfit in outfits:
            for item in outfit['items']:
                set[outfit['set_id']+'_'+str(item['index'])] = item['item_id']
        f2 = open(self.root_dir+'/compatibility_train.txt', 'r')
        X, y = [], []
        a = 0
        lines = f2.readlines()
        for line in lines:
            line = line.split()
            label = int(line[0])
            line = line[1:]
            n = len(line)
            seq = []
            for i in range(n):
                seq.append(set[line[i]])
            X.append(seq)
            y.append(label)
        y = LabelEncoder().fit_transform(y)
        print('len of X: {}'.format(len(X)))
        X_ = tf.keras.preprocessing.sequence.pad_sequences(X,padding='post')
        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, X



class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = tf.stack(X), np.stack(y)
        return tf.transpose(X, [0,1,4,3,2]), tensorflow.keras.utils.to_categorical(y, num_classes=2)


    def __data_generation(self, indexes):
        X = []; y = []
        for idx in indexes:
            seq = []
            for i in range(len(self.X[0])):

                if self.X[idx][i] == 0:
                    seq.append(tf.zeros((3,224,224),dtype=tf.float32))
                else:
                    file_path = osp.join(self.image_dir, str(self.X[idx][i])+'.jpg')
                    seq.append(tf.convert_to_tensor(self.transform(Image.open(file_path)),dtype=tf.float32))

            X.append(tf.stack(seq))
            y.append(self.y[idx])
        return X, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
