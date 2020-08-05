from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Input, MaxPool2D, Dropout, BatchNormalization, Bidirectional, LSTM, Activation
from tensorflow.keras.models import Model
import torch

from data4 import polyvore_dataset, DataGenerator
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


    # inp = Input((19,224,224,3))

    # conv1 = Conv2D(6,kernel_size=3,padding='same')
    # bn1 = BatchNormalization()
    # a1 = Activation('relu')
    # conv2 = Conv2D(12,kernel_size=3,padding='same')
    # bn2 = BatchNormalization()
    # a2 = Activation('relu')
    # mp1 = MaxPool2D()
    # drop1 = Dropout(0.25)

    # conv3 = Conv2D(24,kernel_size=3,padding='same')
    # bn3 = BatchNormalization()
    # a3 = Activation('relu')
    # conv4 = Conv2D(48,kernel_size=3,padding='same')
    # bn4 = BatchNormalization()
    # a4 = Activation('relu')
    # mp2 = MaxPool2D()
    # drop2 = Dropout(0.25)

    # gp = GlobalAveragePooling2D()
    # d1 = Dense(20,activation='relu')


    # lstm = Bidirectional(LSTM(10))
    # d2 = Dense(2,activation='softmax')


    # batch = Config['batch_size']
    # x = inp
    # _, length, row, col, chn = x.shape
    # x = tf.reshape(x, (batch*length, row, col, chn))
    # x = conv1(x)
    # x = bn1(x)
    # x = a1(x)
    # x = conv2(x)
    # x = bn2(x)
    # x = a2(x)
    # x = mp1(x)
    # x = drop1(x)

    # x = conv3(x)
    # x = bn3(x)
    # x = a3(x)
    # x = conv4(x)
    # x = bn4(x)
    # x = a4(x)
    # x = mp2(x)
    # x = drop2(x)

    # x = gp(x)
    # x = d1(x)
    # x = tf.reshape(x, (batch, length, 20))
    # x = lstm(x)
    # x = d2(x)


    # model = Model(inputs=[inp], outputs=x)
    model = tf.keras.models.load_model('seq_model2')

    def scheduler(epoch):
        return 0.0020 * (0.5**(epoch//3))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    saver = tf.keras.callbacks.ModelCheckpoint(filepath='seq_model3',save_best_only=True,verbose=1)

        # define optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model)
    model.summary()

    # training
    history = model.fit(x=train_generator,validation_data=test_generator,epochs=Config['num_epochs'],workers=4,callbacks=[saver,callback])
    plot_training_curve(history.history)
