import sys
sys.path.append(r'a4/keras')
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Input, MaxPool2D, Dropout, BatchNormalization, LSTM, concatenate
from tensorflow.keras.models import Model
import torch

from data3 import polyvore_dataset, DataGenerator
from utils import Config, plot_training_curve
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet





if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # Use GPU


    inp1 = Input((224,224,3))
    inp2 = Input((224,224,3))
    conv1 = Conv2D(6,kernel_size=3,padding='same',activation='relu')
    conv2 = Conv2D(12,kernel_size=3,padding='same',activation='relu')
    bn1 = BatchNormalization()
    mp1 = MaxPool2D()

    conv3 = Conv2D(24,kernel_size=3,padding='same',activation='relu')
    conv4 = Conv2D(48,kernel_size=3,padding='same',activation='relu')
    bn2 = BatchNormalization()
    mp2 = MaxPool2D()
    drop1 = Dropout(0.25)

    conv5 = Conv2D(96,kernel_size=3,padding='same',activation='relu')
    conv6 = Conv2D(192,kernel_size=3,padding='same',activation='relu')
    bn3 = BatchNormalization()
    mp3 = MaxPool2D()

    conv7 = Conv2D(384,kernel_size=3,padding='same',activation='relu')
    bn4 = BatchNormalization()
    mp4 = MaxPool2D()
    drop2 = Dropout(0.25)


    conv11 = Conv2D(6,kernel_size=3,padding='same',activation='relu')
    conv22 = Conv2D(12,kernel_size=3,padding='same',activation='relu')
    bn11 = BatchNormalization()
    mp11 = MaxPool2D()

    conv33 = Conv2D(24,kernel_size=3,padding='same',activation='relu')
    conv44 = Conv2D(48,kernel_size=3,padding='same',activation='relu')
    bn22 = BatchNormalization()
    mp22 = MaxPool2D()
    drop11 = Dropout(0.25)

    gp1 = GlobalAveragePooling2D()
    gp2 = GlobalAveragePooling2D()
    d1 = Dense(20,activation='relu')
    d2 = Dense(20,activation='relu')
    drop2 = Dropout(0.25)
    drop3 = Dropout(0.25)
    lstm = LSTM(10)
    d3 = Dense(2,activation='softmax')



    x1 = inp1
    x2 = inp2
    x1 = conv1(x1)
    x2 = conv11(x2)
    x1 = conv2(x1)
    x2 = conv22(x2)
    x1 = bn1(x1)
    x2 = bn11(x2)
    x1 = mp1(x1)
    x2 = mp11(x2)

    x1 = conv3(x1)
    x2 = conv33(x2)
    x1 = conv4(x1)
    x2 = conv44(x2)
    x1 = bn2(x1)
    x2 = bn22(x2)
    x1 = mp2(x1)
    x2 = mp22(x2)
    x1 = drop1(x1)
    x2 = drop11(x2)


    # base = MobileNet(weights='imagenet', include_top=False)
    # for layer in base.layers:
    #     layer.trainable = False
    # x1 = inp1
    # x2 = inp2
    # for i in range(1,len(base.layers)):
    #     x1 = base.layers[i](x1)
    #     x2 = base.layers[i](x2)

    x1 = gp1(x1)
    x2 = gp2(x2)
    x1 = d1(x1)
    x2 = d2(x2)
    x = tf.stack([x1,x2],axis=1)
    x = lstm(x)
    x = d3(x)
    model = Model(inputs=[inp1,inp2], outputs=x)

    def scheduler(epoch):
        return 0.005 * (0.4**(epoch//3))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    saver = tf.keras.callbacks.ModelCheckpoint(filepath='saved/model',save_best_only=True,verbose=1)

        # define optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model)
    model.summary()

    # training
    history = model.fit(x=train_generator,validation_data=test_generator,epochs=5,workers=4,callbacks=[saver,callback])
    plot_training_curve(history.history)
