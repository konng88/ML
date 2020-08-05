from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import tensorflow as tf
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

        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        for outfit in outfits:
            for item in outfit['items']:
                set[outfit['set_id']+'_'+str(item['index'])] = item['item_id']
        f2 = open(self.root_dir+'/compatibility_train.txt', 'r')

        X, y = [], []
        cat2id = {}
        lines = f2.readlines()

        for line in lines:
            line = line.split()
            label = int(line[0])
            line = line[1:]
            n = len(line)
            for i in range(n):
                if meta_json[set[line[i]]]['category_id'] not in cat2id.keys():
                    cat2id[meta_json[set[line[i]]]['category_id']] = len(cat2id)
            for i in range(n):
                for j in range(i+1,n):
                    X.append((set[line[i]], set[line[j]], cat2id[meta_json[set[line[i]]]['category_id']], cat2id[meta_json[set[line[j]]]['category_id']]))
                    y.append(label)
        y = LabelEncoder().fit_transform(y)
        print('len of X: {}'.format(len(X)))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test



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
        meta = Config['root_path']
        X1, X2, c1, c2, y = self.__data_generation(indexes)
        X1, X2, c1, c2, y = np.stack(X1), np.stack(X2), np.stack(c1), np.stack(c2), np.stack(y)
        return (np.moveaxis(X1, 1, 3), np.moveaxis(X2, 1, 3), c1, c2), tensorflow.keras.utils.to_categorical(y, num_classes=2)


    def __data_generation(self, indexes):
        X1 = []; X2 = []; class1 = []; class2 = []; y = []
        for idx in indexes:
            file_path0 = osp.join(self.image_dir, self.X[idx][0]+'.jpg')
            file_path1 = osp.join(self.image_dir, self.X[idx][1]+'.jpg')
            X1.append(self.transform(Image.open(file_path0)))
            X2.append(self.transform(Image.open(file_path1)))
            class1.append(int(self.X[idx][2]))
            class2.append(int(self.X[idx][3]))
            y.append(self.y[idx])
        return X1, X2, class1, class2, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)