import numpy as np
import pandas as pd
from hw1_dt import TreeNode,DecisionTree
import utils as U

def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sample_under_this_node = 0
    average_entropy_of_child = 0
    for child in branches:
        sample_under_this_node += len(child)
    for child in branches:
        labels,count = np.unique(child, return_counts=True)
        entropy_of_child = 0
        for ith_count in count:
            possibility = ith_count / len(child)
            entropy_of_child += - possibility * np.log2(possibility)
        average_entropy_of_child += (len(child) / sample_under_this_node) * entropy_of_child
    IG = S - average_entropy_of_child
    print(average_entropy_of_child)
    print(IG)

def f1():
    branches = [[1,1],[0,0,0,0],[1,1,1,1,0,0]]
    S=np.log2(2)
    Information_Gain(S,branches)

def f2():
    a=[[1,2,2],[4,5,6],[7,7,7]]
    for i in a:
        val,con =np.unique(i,return_counts=True)
        print(val)
        print(con)

def f3():
    dic ={}
    for i in [1,2,3,4,5,3,2,1,1,1,1]:
        if i not in dic.keys():
            dic[i] = [1]
        else:
            dic[i].append(1)
    print(dic)
    print(list(dic.values()))


def f4():
    a=[1,2,3,4,5,3,2,1,1,1,1]
    b=np.unique(a,return_counts=True)[1]
    print(b)

def f6():
    features = [[1,0,1],[2,2,1],[1,3,1],[2,2,1],[2,0,1],[1,1,1],[0,3,1],[1,2,1],[2,3,1],[2,1,1],[0,2,1],[2,3,1]]
    # featuresT = np.array(features)
    # D = len(featuresT)
    # featuresT=np.delete(featuresT,0,axis=1)
    # featuresT=np.delete(featuresT,0,axis=1)
    # featuresT=np.delete(featuresT,0,axis=1)
    # features = featuresT.tolist()
    # print(features)
    feature = features[0]
    f = np.delete(feature,0,axis = 0)
    p = feature.pop(0)
    print(f)
    print(p)
def f7():
    features = [[1,0,1],[2,2,1],[1,3,1],[2,2,1],[2,0,1],[1,1,1],[0,3,1],[1,2,1],[2,3,1],[2,1,1],[0,2,1],[2,3,1]]
    e = enumerate(features)
    for i,f in e:
        print(i,f)
def p(node):
    if node.splittable == False:
        print(node.cls_max)
        return
    print('-------')
    for c in node.children:
        p(c)
    print('-------')


def f8():
    a = [1,2,3,5,4,2,1,3,5,8,4,1,2,3]
    u = [1,0]
    u1 = np.unique(a)
    print(u1)
    print(int(np.argwhere(u1 == 3)))
    print(u)
    print(u.index(0))

def f9():
    a = [0,1,2,3,4]
    b = [5,1,3,4,2]
    c=sorted(a,key = lambda x:b[x])
    print(c)

def f10():
    a = [(1,2),(2,3),(3,4),(1,2),(5,8),(3,2)]
    b=a[:3]
    s = []
    for i in b:
        s.append(i[0])
    print(s)

def f11():
    # data=np.loadtxt('car.data',delimiter=',')
    # data = pd.DataFrame(data = data)
    # labels = data[0].tolist()
    # features = np.array(data.drop([0])).tolist()
    features = [[1,0,1],[2,2,1],[1,3,1],[2,2,1],[2,0,1],[1,1,1],[0,3,1],[1,2,1],[2,3,1],[2,1,1],[0,2,1],[2,3,1]]
    labels = [1,0,1,1,0,1,0,1,0,0,0,1]
    root = TreeNode(features=features,labels=labels,num_cls=3)
    root.split()
    # p(root)

    # for i in range(0,3):
    #     print(root.children[i].cls_max,root.children[i].dim_split)
    # for i in range(0,4):
    #     print(root.children[1].children[i].cls_max)

    # for i in range(0,3):
    #     for j in range(0,3):
    #         for k in range(0,3):
    #             print(root.predict([i,j,k]))
    print(root.predict([0,1,1]))
    print('---')
    print(root.predict([1,1,1]))
    print('---')
    print(root.predict([2,1,1]))

def t1():
    data = np.loadtxt('car.data',delimiter=',')
    x_train = pd.DataFrame(data)
    y_train = x_train[['0']].tolist()
    x_train = x_train.drop([['0']],axis=1)
    x_train = np.array(x_train).tolist()
    tree = DecisionTree()
    tree.train(x_train,y_train)
    p = tree.predict(x_train)
    e = 0
    for i in range(0,len(p)):
        if y_train[i] != p[i]:
            print(y_train[i],p[i])
            e += 1
    print(e/i)


def t2():
    data = np.loadtxt('car.data',delimiter=',')
    x_train = pd.DataFrame(data)
    y_train = x_train[0].tolist()
    x_train = x_train.drop([0],axis=1)
    x_train = np.array(x_train).tolist()
    x_test = x_train[1500:]
    y_test = y_train[1500:]
    x_train = x_train[:1500]
    y_train = y_train[:1500]

    tree = DecisionTree()
    tree.train(x_train,y_train)
    p = tree.predict(x_train)
    U.print_tree(decisionTree=tree)
    U.reduced_error_prunning(decisionTree=tree,X_test=x_test,y_test=y_test)
    print('---------------------------')
    U.print_tree(decisionTree=tree)

def t3():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
    labels = [0, 0, 1, 1]
    tree = DecisionTree()
    tree.train(features,labels)
    root = tree.root_node
    p(root)
    features = [['a', 'b'], ['b', 'a'], ['b', 'c']]
    print(tree.predict(features))

def f12():
    a = np.arange(0,10).tolist()
    print(a)
    a.remove(8)
    print(a)

def f13():
    a = [1,2,3,3,2,5,4,6,5,8,4,1,2,3,7]
    bb,b = np.unique(a,return_counts=True)
    print(bb)
    print(b)

def f14():
    a = [1,2,3,4,5,6]
    print(sum(a))

def t4():
    features = [[0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 2, 1, 0],
                [2, 2, 1, 1],
                [1, 2, 1, 1],
                [0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]
                ]
    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    tree = DecisionTree()
    tree.train(features,labels)
    root = tree.root_node
    def print_tree_info(node):
        print('-----------')
        print('features: ',node.features)
        print('labels: ',node.labels)
        print('dim_split: ',node.dim_split)
        print('-----------')
        for child in node.children:
            print_tree_info(child)
    print_tree_info(root)


def qt():
    x = [[1,2,3,4,5,6,7,8,9]]
    x = np.array(x)
    print(x)
    y = x.T
    print(y)

def f15():
    a = [1,5,8,3,6]
    a.sort()
    print(a)

f15()
