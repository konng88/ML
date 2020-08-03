import numpy as np
import h5py
import sys
import os
import argparse
import json
import warnings
from matplotlib import pyplot as plt
# warnings.filterwarnings('ignore')
np.random.seed(98)

class linear_layer:

    def __init__(self, input_D, output_D,initializer='normal'):
        self.input_D = input_D
        self.output_D = output_D
        self.params = dict()
        if initializer == 'normal':
            self.params['W'] = np.random.normal(size=(input_D,output_D),loc=0,scale=0.1)
            self.params['b'] = np.random.normal(size=(1,output_D),loc=0,scale=0.1)
        elif initializer == 'uniform':
            self.params['W'] = np.random.uniform(size=(input_D,output_D),low=-0.5,high=0.5)
            self.params['b'] = np.random.uniform(size=(1,output_D),low=-0.5,high=0.5)
        elif initializer == 'ones':
            self.params['W'] = np.ones((input_D,output_D))/100
            self.params['b'] = np.ones((1,output_D))/100
        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D,output_D))
        self.gradient['b'] = np.zeros((1,output_D))

    def forward(self, X):
        forward_output = np.dot(X,self.params['W']) + self.params['b']
        return forward_output

    def backward(self, X, grad):
        self.gradient['W'] = np.dot(X.T,grad)
        self.gradient['b'] = np.dot(np.ones((1,len(X))),grad)
        backward_output = np.dot(grad,self.params['W'].T)
        return backward_output

class relu:

    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = np.array(X > 0).astype(float)
        forward_output = np.multiply(self.mask,X)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(self.mask,grad)
        return backward_output

class tanh:

    def forward(self, X):
        forward_output = np.tanh(X)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply((np.ones(X.shape) - np.tanh(X)**2),grad)
        return backward_output

class dropout:

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train=False):
        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(self.mask,grad)
        return backward_output

class softmax_cross_entropy:

    def __init__(self):
        self.grad = None

    def forward(self, X, y):
        mxs = np.dot(np.max(X,axis=1).reshape((len(X),1)),np.ones((1,len(X[0]))))
        c = np.exp(X-mxs)
        sum = np.dot(np.sum(c,axis=1).reshape((len(X),1)),np.ones((1,len(X[0]))))
        self.grad = c/sum
        return -np.sum(y*np.log(self.grad))

    def backward(self,X, y):
        return self.grad - y

def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum

def miniBatchStochasticGradientDescent(model, momentum, _lambda, _alpha, _learning_rate):
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                if _alpha > 0.0:
                    g = module.gradient[key] + _lambda * module.params[key]
                    M_key = module_name + '_' + key
                    momentum[M_key] = momentum[M_key] * _alpha - _learning_rate * g
                    module.params[key] = module.params[key] + momentum[M_key]
                else:
                    module.params[key] = module.params[key] - _learning_rate  * module.gradient[key]
    return model

def predict_label(f):
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        y_predict = np.zeros(f.shape)
        for i in range(len(f)):
            y_predict[i][np.argmax(f[i])] = 1
        return 2*y_predict-1

def main(main_params, optimization_type="minibatch_sgd"):
    np.random.seed(int(main_params['random_seed']))
    trainFile = main_params['train_file']
    testFile = main_params['test_file']
    is_test = main_params['is_test']
    with h5py.File('mnist_traindata.hdf5') as f:
        if is_test:
            Xtrain = f['xdata'][:]/255
            Ytrain = f['ydata'][:]
        else:
            Xtrain = f['xdata'][:50001]/255
            Ytrain = f['ydata'][:50001]
        Xval = f['xdata'][50000:]/255
        Yval = f['ydata'][50000:]
    with h5py.File('mnist_testdata.hdf5') as f:
        Xtest = f['xdata'][:]/255
        Ytest = f['ydata'][:]
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape

    index = np.arange(10)
    unique, counts = np.unique(Ytrain, return_counts=True)
    counts = dict(zip(unique, counts)).values()
    model = dict()

    num_L2 = 10
    _initializer = main_params['initializer']
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])
    _learning_rate = float(main_params['learning_rate'])
    num_L1 = int(main_params['num_neurons'])
    _step = int(main_params['decay_every'])
    _alpha = float(main_params['alpha'])
    _lambda = float(main_params['lambda'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']
    _savefig = main_params['savefig']

    if _activation == 'relu':
        act = relu
    else:
        act = tanh


    model['L1'] = linear_layer(input_D = d, output_D = num_L1, initializer=_initializer)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r = _dropout_rate)
    model['L2'] = linear_layer(input_D = num_L1, output_D = num_L2, initializer=_initializer)
    model['loss'] = softmax_cross_entropy()

    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []
    decay_epochs = []

    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            decay_epochs.append(t)
            _learning_rate = _learning_rate * 0.5

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0
        indices = np.random.permutation(N_train)
        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = Xtrain[indices[i*minibatch_size:(i+1)*minibatch_size]], Ytrain[indices[i*minibatch_size:(i+1)*minibatch_size]]
            ### forward ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = True)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            ### backward ###
            grad_a2 = model['loss'].backward(a2, y)
            grad_d1 = model['L2'].backward(d1,grad_a2)
            grad_h1 = model['drop1'].backward(h1,grad_d1)
            grad_a1 = model['nonlinear1'].backward(a1,grad_h1)
            grad_x = model['L1'].backward(x, grad_a1)

            model = miniBatchStochasticGradientDescent(model, momentum, _lambda, _alpha, _learning_rate)

        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = Xtrain[indices[i*minibatch_size:(i+1)*minibatch_size]], Ytrain[indices[i*minibatch_size:(i+1)*minibatch_size]]

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = False)
            a2 = model['L2'].forward(d1)

            loss = model['loss'].forward(a2, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_loss = train_loss
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        indices = np.arange(N_val)
        for i in range(int(np.floor(N_val / minibatch_size))):

            x, y = Xval[indices[i*minibatch_size:(i+1)*minibatch_size]], Yval[indices[i*minibatch_size:(i+1)*minibatch_size]]


            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train = False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)
        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    # save file
    name = 'MLP_lr' + str(main_params['learning_rate']) + '_m' + str(main_params['alpha']) + '_w' + str(main_params['lambda']) + '_d' + str(main_params['dropout_rate']) + '_a' + str(main_params['activation']) + '_b' + str(main_params['minibatch_size']) + '_n' + str(main_params['num_neurons'])
    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open(name + '.json', 'w'))
    xrange = np.linspace(0,main_params['num_epoch'],main_params['num_epoch'])
    plt.plot(xrange,train_acc_record,color='blue',label='Train Accuracy')
    plt.plot(xrange,val_acc_record,color='red',label='Validation Accuracy')
    for e in decay_epochs:
        plt.vlines(e,min(train_acc_record[0],val_loss_record[0]),1,color='grey',label='Learning Rate Decay')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy' + name)
    if _savefig:
        plt.savefig('Accuracy' + name + '.png')
    plt.show()
    plt.plot(xrange,train_loss_record,color='green',label='Train Loss')
    plt.plot(xrange,val_loss_record,color='yellow',label='Validation Loss')
    for e in decay_epochs:
        plt.vlines(e,0,max(val_loss_record[0],train_loss_record[0]),color='grey',label='Learning Rate Decay')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss' + name)
    if _savefig:
        plt.savefig('Loss' + name + '.png')
    plt.show()

    # test
    if is_test:
        a1 = model['L1'].forward(Xtest)
        h1 = model['nonlinear1'].forward(a1)
        d1 = model['drop1'].forward(h1, is_train = False)
        a2 = model['L2'].forward(d1)
        test_loss = model['loss'].forward(a2, Ytest)
        test_acc = np.sum(predict_label(a2) == Ytest)/len(Ytest)


    # Summary
    print()
    print('-'*10,'Training Summary','-'*10)
    print('Train Loss:',train_loss)
    print('Train Accuracy:',train_acc)
    print('Validation Loss:',val_loss)
    print('Validation Accuracy:',val_acc)
    if is_test:
        print('-'*10,'Testing Summary','-'*10)
        print('Test Loss:',test_loss)
        print('Test Accuracy:',test_acc)
    print()
    print('Finish running!')
    return train_loss_record, val_loss_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--initializer', default='ones')
    parser.add_argument('--learning_rate', default=0.0005)
    parser.add_argument('--decay_every', default=18)
    parser.add_argument('--alpha', default=0.95)
    parser.add_argument('--lambda', default=0.0001)
    parser.add_argument('--dropout_rate', default=0.2)
    parser.add_argument('--num_epoch', default=50)
    parser.add_argument('--minibatch_size', default=50)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--num_neurons', default=100)
    parser.add_argument('--train_file', default='mnist_traindata.hdf5')
    parser.add_argument('--test_file', default='mnist_testdata.hdf5')
    parser.add_argument('--savefig', default=False)
    parser.add_argument('--is_test', default=True)
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)
