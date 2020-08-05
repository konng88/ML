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


    def scheduler(epoch):
        return 0.0008 * (0.6**(epoch//2))

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    saver = tf.keras.callbacks.ModelCheckpoint(filepath='savedmodel2',save_best_only=True,verbose=1)

        # define optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model)
    model.summary()

    # training
    history = model.evaluate(x=test_generator)
    print(history)
