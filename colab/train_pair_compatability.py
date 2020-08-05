from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Input, MaxPool2D, Dropout, BatchNormalization, LSTM, Activation, Embedding, concatenate
from tensorflow.keras.models import Model
import torch

from data2 import polyvore_dataset, DataGenerator
from utils import Config, plot_training_curve
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet





if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test = dataset.create_dataset()


    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # Use GPU


    inp1 = Input((224,224,3))
    inp2 = Input((224,224,3))

    conv1 = Conv2D(6,kernel_size=3,padding='same')
    bn1 = BatchNormalization()
    a1 = Activation('relu')
    conv2 = Conv2D(12,kernel_size=3,padding='same')
    bn2 = BatchNormalization()
    a2 = Activation('relu')
    mp1 = MaxPool2D()

    conv3 = Conv2D(24,kernel_size=3,padding='same')
    bn3 = BatchNormalization()
    a3 = Activation('relu')
    conv4 = Conv2D(48,kernel_size=3,padding='same')
    bn4 = BatchNormalization()
    a4 = Activation('relu')
    mp2 = MaxPool2D()
    drop1 = Dropout(0.25)


    conv11 = Conv2D(6,kernel_size=3,padding='same')
    bn11 = BatchNormalization()
    a11 = Activation('relu')
    conv22 = Conv2D(12,kernel_size=3,padding='same')
    bn22 = BatchNormalization()
    a22 = Activation('relu')
    mp11 = MaxPool2D()

    conv33 = Conv2D(24,kernel_size=3,padding='same')
    bn33 = BatchNormalization()
    a33 = Activation('relu')
    conv44 = Conv2D(48,kernel_size=3,padding='same')
    bn44 = BatchNormalization()
    a44 = Activation('relu')
    mp22 = MaxPool2D()
    drop11 = Dropout(0.25)

    gp1 = GlobalAveragePooling2D()
    gp2 = GlobalAveragePooling2D()
    d1 = Dense(20,activation='relu')
    d2 = Dense(20,activation='relu')
    lstm = LSTM(10)
    d3 = Dense(2,activation='softmax')

    inp3 = Input(1)
    inp4 = Input(1)
    emb = Embedding(input_dim=153, output_dim=10)


    x1 = inp1
    x2 = inp2
    x1 = conv1(x1)
    x2 = conv11(x2)
    x1 = bn1(x1)
    x2 = bn11(x2)
    x1 = a1(x1)
    x2 = a11(x2)
    x1 = conv2(x1)
    x2 = conv22(x2)
    x1 = bn2(x1)
    x2 = bn22(x2)
    x1 = a2(x1)
    x2 = a22(x2)
    x1 = mp1(x1)
    x2 = mp11(x2)

    x1 = conv3(x1)
    x2 = conv33(x2)
    x1 = bn3(x1)
    x2 = bn33(x2)
    x1 = a3(x1)
    x2 = a33(x2)
    x1 = conv4(x1)
    x2 = conv44(x2)
    x1 = bn4(x1)
    x2 = bn44(x2)
    x1 = a4(x1)
    x2 = a44(x2)
    x1 = mp2(x1)
    x2 = mp22(x2)
    x1 = drop1(x1)
    x2 = drop11(x2)

    x3 = inp3
    x4 = inp4
    x3 = emb(x3)
    x4 = emb(x4)
    x3 = tf.reshape(x3,(-1,10))
    x4 = tf.reshape(x4,(-1,10))

    x1 = gp1(x1)
    x2 = gp2(x2)
    x1 = d1(x1)
    x2 = d2(x2)
    x1 = concatenate([x1,x3])
    x2 = concatenate([x2,x4])
    x = tf.stack([x1,x2],axis=1)
    x = lstm(x)
    x = d3(x)
    model = Model(inputs=[inp1,inp2,inp3,inp4], outputs=x)

    def scheduler(epoch):
        return 0.005 * (0.6**(epoch//2))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    saver = tf.keras.callbacks.ModelCheckpoint(filepath='savedmodel',save_best_only=True,verbose=1)

        # define optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model)
    model.summary()

    # training
    history = model.fit(x=train_generator,validation_data=test_generator,epochs=10,workers=4,callbacks=[saver,callback])
    plot_training_curve(history.history)
