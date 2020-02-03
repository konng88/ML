import numpy as np

def f1():
    a=np.random.randint(9,size=(5,3))
    print(a)
    b = [0,2,3]
    print()
    print(a[b])

def f2():
    a=np.random.randint(9,size=(5,3))
    print(a)
    b = [0,2,3]
    print()
    a[1] = a[1] +np.array([1,1,1])
    print(a)


def f3():
    a=np.random.randint(5,size=20)
    print(a)
    print()
    print(np.unique(a,return_counts=True))

def f4():
    a=np.random.randint(9,size=(5,3))
    b=np.random.randint(3,size=5)
    print(a)
    print()
    print(b)
    print()
    print((a.T/b).T)



def f5():
    D=50
    k=8
    a=np.random.randint(8,size=D)
    print(a)
    b=np.dot(np.ones((k,1)),a.reshape((1,D)))
    print(b)
    c=np.random.randint(5,size=(k,D))
    d=np.linalg.norm(b-c,ord=2,axis=1)
    print(c)
    print(d)
    print(d[np.argmax(d)])



def f6():
    a=np.random.randint(9,size = (3,4,5))
    print(a)

def f7():
    a=np.random.randint(9,size = (3,4,5))
    b=np.random.randint(9,size = (3,4,5))
    if a==b:
        print('q')

def f8():
    a=np.random.randint(5,size=(2,3))
    b=np.random.randint(5,size=(3,))
    print(a)
    print()
    print(b)
    print()
    print(b-a)

def f9():
    a=np.array([9,12])
    print(np.linalg.norm(a,ord=2)**2)


def f10():
    a=np.random.randint(9,size=(3,8))
    print(a)
    b=np.ones((2,8))
    print()
    print(b)
    print()
    print(a-b)


def f11():
    f=[1]*5
    print(f)

f11()
