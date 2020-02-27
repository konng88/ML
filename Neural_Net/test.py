import numpy as np

def f1():
    class a:
        x=0
        y=0
        z=True

    w=a()
    print(hasattr(a,'t'))
    print(hasattr(a,'x'))
    print(hasattr(a,'y'))
    print(hasattr(a,'s'))

def f2():
    a=dict()
    a['s']=1
    print(a)


def f3():
    x=np.random.normal(size=(3,4))
    print(x)


def f4():
    W=np.random.randint(9,size=(2,4))
    b=np.random.randint(3,size=4)
    print(W)
    print(b)
    print(np.vstack((W,b)))

def f5():
    X=np.random.randint(9,size=(2,4))
    X_=np.tanh(X)
    print(X_)

def f6():
    N=2
    D=4
    newD=3
    X=np.random.randint(9,size=(N,D))
    W=np.random.randint(9,size=(D,newD))
    b=np.random.randint(9,size=(1,newD))
    X_=np.hstack((X,np.ones((N,1))))
    W_=np.vstack((W,b))
    print(np.dot(X_,W_))
    B=np.dot(np.ones((N,1)),b)
    print(np.dot(X,W)+B)


def f7():
    W=np.random.randint(9,size=(2,4))
    print(W)
    print()
    print(W-1)


def f8():
    a=np.random.randint(7,size = (2,4))
    b=np.ones((1,4))
    print(a)
    print(a+b)


def f9():
    a=np.random.normal(size=(1,4))
    print(a)
    print(a.mean())
    print(a.var())

def f10():
    a=np.random.normal(size=(2,4))
    b=np.eye(2.5)
    print(b)

def f11():
    a = np.random.randint(10,size=(2,3))
    print(a)
    mx = np.dot(np.max(a,axis=1).reshape((len(a),1)),np.ones((1,len(a[0]))))
    print(mx)
    c = np.exp(a-mx)
    print(c)
    sum = np.dot(np.sum(c,axis=1).reshape((len(a),1)),np.ones((1,len(a[0]))))
    print(sum)
    e = c/sum
    print(e)
f11()
