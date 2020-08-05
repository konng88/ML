from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, concatenate, Dropout
from tensorflow.keras.models import Model
import torch

from data3 import polyvore_dataset, DataGenerator
from utils import Config, plot_training_curve

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2





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

    base1 = MobileNetV2(weights='imagenet', include_top=False)
    base2 = MobileNetV2(weights='imagenet', include_top=False)
    for layer in base1.layers:
        layer.trainable = False
    for layer in base2.layers:
        layer.trainable = False
        layer._name = layer.name+'_22'

    x1 = base1.output
    x2 = base2.output
    gp1 = GlobalAveragePooling2D()
    gp2 = GlobalAveragePooling2D()
    drop = Dropout(0.25)
    d1 = Dense(512,activation='relu')
    d2 = Dense(2,activation='softmax')
    x1 = gp1(x1)
    x2 = gp2(x2)
    x = concatenate([x1,x2])
    x = drop(x)
    x = d1(x)
    x = d2(x)


    model = Model(inputs=[base1.input,base2.input], outputs=x)

    def scheduler(epoch):
        return 0.001 * (0.5**(epoch//5))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # define optimizers
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model)
    # tf.keras.utils.plot_model(model)
    model.summary()
    model.save('model.h5py')
    # training
    history = model.fit(x=train_generator,validation_data=test_generator,epochs=Config['num_epochs'],workers=2,callbacks=[callback])
    plot_training_curve(history.history)
