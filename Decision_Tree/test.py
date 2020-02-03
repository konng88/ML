import numpy as np
import utils as Util
from hw1_dt import DecisionTree,TreeNode

def f():
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
    print_info(root)


def print_info(node):
    if node.splittable == True:
        print('---------------')
        print('dim_split: ',node.dim_split)
        print('features: ',node.features)
        print('labels: ',node.labels)
        print('num_children: ',len(node.children))
        if node.children != []:
            for i in range(0,len(node.children)):
                print(i,'th children features: ',node.children[i].features)
                print(i,'th children labels: ',node.children[i].labels)
        print('---------------')
    for child in node.children:
        print_info(child)


f()
