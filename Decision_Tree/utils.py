import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sample_under_this_node = 0
    average_entropy_of_child = 0
    for child in branches:
        for i in child:
            sample_under_this_node += i
    for child in branches:
        entropy_of_child = 0
        sample_in_child = 0
        for i in child:
            sample_in_child += i
        for i in child:
            if sample_in_child == 0:
                entropy_of_child = 0
            else:
                possibility = i / sample_in_child
                if possibility != 0:
                    entropy_of_child += - possibility * np.log2(possibility)
        average_entropy_of_child += (sample_in_child / sample_under_this_node) * entropy_of_child
    IG = S - average_entropy_of_child
    return IG
    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    root = decisionTree.root_node
    def bottom_up_prunning(node):
        if node.splittable == True:
            for child_node in node.children:
                bottom_up_prunning(child_node)
            prediction_before = decisionTree.predict(X_test)
            error_before = 0
            for i in range(0,len(y_test)):
                if prediction_before[i] != y_test[i]:
                    error_before +=1
            node.splittable = False
            prediction_after = decisionTree.predict(X_test)
            error_after = 0
            for i in range(0,len(y_test)):
                if prediction_after[i] != y_test[i]:
                    error_after +=1
            if error_before < error_after:
                node.splittable = True
    bottom_up_prunning(root)

    # raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
