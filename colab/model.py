from data import polyvore_dataset, DataGenerator
from utils import Config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Layer, Conv2D, MaxPool2D, Dense, Dropout, LeakyReLU, Flatten, Activation, BatchNormalization, UpSampling2D, Conv2DTranspose, Reshape, concatenate

def my_model():
    input1 = Input(shape=(224,224,3))
    conv1 = Conv2D(10,kernel_size=3,padding='same',activation='relu')(input1)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(40,kernel_size=3,padding='same',activation='relu')(bn1)
    bn2 = BatchNormalization()(conv2)
    mp1 = MaxPool2D()(bn2)
    d1 = Dropout(0.25)(mp1)

    conv3 = Conv2D(100,kernel_size=3,padding='same',activation='relu')(d1)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv2D(400,kernel_size=3,padding='same',activation='relu')(bn3)
    bn4 = BatchNormalization()(conv4)
    mp2 = MaxPool2D()(bn4)
    d2 = Dropout(0.25)(mp2)

    # conv5 = Conv2D(128,kernel_size=3,padding='same',activation='relu')(d2)
    # bn5 = BatchNormalization()(conv5)
    # conv6 = Conv2D(256,kernel_size=3,padding='same',activation='relu')(bn5)
    # bn6 = BatchNormalization()(conv6)
    # mp3 = MaxPool2D()(bn6)
    # d3 = Dropout(0.25)(mp3)
    #
    # conv7 = Conv2D(512,kernel_size=3,padding='same',activation='relu')(d3)
    # bn7 = BatchNormalization()(conv7)
    # conv8 = Conv2D(1024,kernel_size=3,padding='same',activation='relu')(bn7)
    # bn8 = BatchNormalization()(conv8)
    # mp4 = MaxPool2D()(bn8)
    # d4 = Dropout(0.25)(mp4)
    #
    # conv9 = Conv2D(2048,kernel_size=3,padding='same',activation='relu')(d4)
    # bn9 = BatchNormalization()(conv9)
    # conv10 = Conv2D(4096,kernel_size=3,padding='same',activation='relu')(bn9)
    # bn10 = BatchNormalization()(conv10)
    # mp5 = MaxPool2D()(bn10)
    # d5 = Dropout(0.25)(mp5)

    model = keras.Model(inputs=[input1],outputs=[d5])
    return model


class ConvBlock(Layer):
    """
    @ filters = first Conv layer filters
    @ output channels = 2 * filters
    """
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.filters = filters
        tf.keras.backend.set_floatx('float32')
    def build(self,input_shape):
        self.conv1 = Conv2D(self.filters, kernel_size=3, padding='same',activation='tanh')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(self.filters*2, kernel_size=3, padding='same',activation='tanh')
        self.bn2 = BatchNormalization()
        self.mp = MaxPool2D()
    def call(self, X):
        X = tf.cast(X, tf.float32)
        o = self.conv1(X)
        o = self.bn1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.mp(o)
        return o
class ImgEmb(Layer):
    def __init__(self):
        super(ImgEmb, self).__init__()
        tf.keras.backend.set_floatx('float32')
    def build(self, input_shape):
        self.cb1 = ConvBlock(6)
        self.cb2 = ConvBlock(24)
        self.cb3 = ConvBlock(96)
        self.cb4 = ConvBlock(384)
        self.cb5 = ConvBlock(1536)
        self.gp = keras.layers.GlobalAveragePooling2D()
        self.d = Dense(50,activation='tanh')

    def call(self, X):
        o = self.cb1(X)
        o = self.cb2(o)
        o = self.cb3(o)
        o = self.cb4(o)
        o = self.cb5(o)
        o = self.gp(o)
        o = self.d(o)
        return o


class Comp_model(Layer):
    def __init__(self):
        super(Comp_model, self).__init__()
        tf.keras.backend.set_floatx('float32')
    def build(self,input_shape):
        b,l,r,c,cn = input_shape
        self.leng = input_shape[1]
        self.re1 = Reshape((r,c,cn))
        self.emb = ImgEmb()
        self.re2 = Reshape((-1,50))
        self.rnn = LSTM(50)
        self.d = Dense(1)
    def call(self, X):
        batch, leng, row, col, chn = X.shape
        o = tf.cast(X, tf.float32)
        o = self.re1(o)
        print(o.shape)
        o = self.emb(o)
        o = self.re2(o)
        o = self.rnn(o)
        o = self.d(o)
        return o
    # def reshape1(self,x):
    #     print(x.shape)
    #     b, l, r, c, c = x.shape
    #     return K.reshape(x, (b*l, 224, 224,3))
    # def reshape2(self,x):
    #     b = x.shape[0]
    #     return K.reshape(x, (b/self.leng, self.leng, 224, 224,3))


# def my_model():
#     input = Input(shape=(224,224,3))
#     conv1 = Conv2D(6,kernel_size=3,padding='same',activation='relu')(input)
#     bn1 = BatchNormalization()(conv1)
#     conv2 = Conv2D(12,kernel_size=3,padding='same',activation='relu')(bn1)
#     bn2 = BatchNormalization()(conv2)
#     mp1 = MaxPool2D()(bn2)
#     d1 = Dropout(0.25)(mp1)
#
#     conv3 = Conv2D(32,kernel_size=3,padding='same',activation='relu')(d1)
#     bn3 = BatchNormalization()(conv3)
#     conv4 = Conv2D(64,kernel_size=3,padding='same',activation='relu')(bn3)
#     bn4 = BatchNormalization()(conv4)
#     mp2 = MaxPool2D()(bn4)
#     d2 = Dropout(0.25)(mp2)
#
#     conv5 = Conv2D(128,kernel_size=3,padding='same',activation='relu')(d2)
#     bn5 = BatchNormalization()(conv5)
#     conv6 = Conv2D(256,kernel_size=3,padding='same',activation='relu')(bn5)
#     bn6 = BatchNormalization()(conv6)
#     mp3 = MaxPool2D()(bn6)
#     d3 = Dropout(0.25)(mp3)
#
#     conv7 = Conv2D(512,kernel_size=3,padding='same',activation='relu')(d3)
#     bn7 = BatchNormalization()(conv7)
#     conv8 = Conv2D(1024,kernel_size=3,padding='same',activation='relu')(bn7)
#     bn8 = BatchNormalization()(conv8)
#     mp4 = MaxPool2D()(bn8)
#     d4 = Dropout(0.25)(mp4)
#
#     conv9 = Conv2D(2048,kernel_size=3,padding='same',activation='relu')(d4)
#     bn9 = BatchNormalization()(conv9)
#     conv10 = Conv2D(4096,kernel_size=3,padding='same',activation='relu')(bn9)
#     bn10 = BatchNormalization()(conv10)
#     mp5 = MaxPool2D()(bn10)
#     d5 = Dropout(0.25)(mp5)
#
#     model = keras.Model(inputs=[input],outputs=[d5])
#     return model


def main(params):
    data = polyvore_dataset()
    transforms = data.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = data.create_dataset()
    train_set = (X_train, y_train, transforms['train'])
    test_set = (X_test, y_test, transforms['test'])
    dataset_size = {'train': len(y_train), 'test': len(y_test)}
    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }
    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


if __name__ == '__main__':


    b,l,r,c,cn = 100,2,224,224,3
    input1 = np.random.randn(b,l,r,c,cn)
    print(input1.shape)

    image_input = Input(shape=(l,r,c,cn) , name='input_img')
    m = Comp_model()(image_input)
    model = keras.Model(inputs=[image_input],outputs=[m])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
