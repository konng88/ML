from utils import Distances
import numpy as np
def f1():
    dict={1:3,2:2,3:5}
    d=sorted(dict.items(),key=lambda x:x[1])
    print(dict)
    print(d)
    return


def f2(k,feature):
    data=[[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4]]
    distances={}
    i=0
    for ith_point in data:
        dis=Distances()
        distances[i] = dis.euclidean_distance(feature,ith_point)
        i += 1
    print(distances)
    d=sorted(distances.items(),key=lambda x:x[1])
    print(d)
    return


def f3():
    dict={88:9,1:3,2:2,3:5,48:6,36:5}
    d=sorted(dict.items(),key=lambda x:x[1])
    print(d)
    for value in d:
        print(value[0])

def f4():
    a=[1,2,3,3,3,8,8,8,8,3,3,5,4,1,2,1,8,5,3,8,8,8,8,8,8,8,8,2,3]
    counts=np.bincount(a)
    m=np.argmax(counts)
    print(counts)
    print(m)

def f5():
    for i in range(1,10,2):
        print(i)

def f6():
    dict={88:9,1:3,2:2,3:5,48:6,36:5,1:7}
    print(list(dict.items()))
    print(list(dict.items())[:3])

def f7():
    a={}
    a[[1,2,3,4,5]]=1

def f8(features):
    """
    Normalize features for every sample

    Example
    features = [[3, 4], [1, -1], [0, 0]]
    return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

    :param features: List[List[float]]
    :return: List[List[float]]
    """
    new_features = []
    for ith_vector in features:
        zero_vector = True
        for ith_num in ith_vector:
            if ith_num != 0:
                zero_vector = False
        if zero_vector == True:
            new_vector = ith_vector
        else:
            sum = 0
            for ith_num in ith_vector:
                sum += ith_num ** 2
            sum = sum ** (1 / 2)
            new_vector = []
            for ith_num in ith_vector:
                new_num = ith_num / sum
                new_vector.append(new_num)
        new_features.append(new_vector)
    return new_features

def f9():
    m=np.array([[2, -1], [-1, 5], [0, 0]])
    l=m.tolist()
    print(l)
    t=m.T
    print(t.tolist())

def f10(point1, point2):
    """
   :param point1: List[float]
   :param point2: List[float]
   :return: float
   """
    x1=np.array(point1)
    x2=np.array(point2)

    print(np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2)))
    D = len(point1)
    point1_length = 0.0
    point2_length = 0.0
    inner_product = 0.0
    for i in range(0,D):
        inner_product += point1[i] * point2[i]
        point1_length += point1[i] ** 2
        point2_length += point2[i] ** 2
    point1_length = np.sqrt(point1_length)
    point2_length = point2_length ** (0.5)
    cosine_similarity_distance = inner_product / (point1_length * point2_length)
    print(float(cosine_similarity_distance))



def f(function):
    function()

class C():
    def __init__(self):
        pass
    def cf1(self):
        print('1')
    def cf2(self):
        self.cf1()

def f11():
    a=[1,2,3,4,5,6,6,5,4,1,2,3]
    print(np.argmax(a))
f11()
