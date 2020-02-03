import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        if self.splittable:
            featuresT = np.array(self.features).T.tolist()
            D = len(featuresT)
            N = len(self.features)
            # calculate entropy
            count = np.unique(self.labels,return_counts = True)[1]
            entropy = 0
            for i in count:
                possibility = i / N
                if possibility != 0:
                    entropy -= possibility * np.log2(possibility)
            # split by feature d
            def split_by(d):
                this_feature = featuresT[d]
                label_dic = {}
                feature_dic = {}
                entropy = 0
                for n in range(0,N):
                    point = self.features[n]
                    label = self.labels[n]
                    if this_feature[n] not in label_dic.keys():
                        label_dic[this_feature[n]] = [label]
                        feature_dic[this_feature[n]] = [point]
                    else:
                        label_dic[this_feature[n]].append(label)
                        feature_dic[this_feature[n]].append(point)
                branches_features = list(feature_dic.values())
                branches_labels = list(label_dic.values())
                branches_count = []
                for branch_labels in branches_labels:
                    branch_count = np.unique(branch_labels,return_counts = True)[1].tolist()
                    branches_count.append(branch_count)
                return branches_features,branches_labels,branches_count

                # greed best feature
            dic_IG = {}   # dic_IG = { d : IG }
            d_num_attributes = {}
            for d in range(0,D):
                branches_features,branches_labels,branches_count = split_by(d)
                IG = Util.Information_Gain(entropy,branches_count)
                dic_IG[d] = IG
                d_num_attributes[d] = len(branches_features)
            sorted_IG = sorted(dic_IG.items(),key = lambda x: x[1],reverse=True)  # sorted_IG = [(d , IG)]
            num_tie = 1
            for i in range(0,len(sorted_IG)-1):
                if sorted_IG[i][1] != sorted_IG[i+1][1]:
                    break
                num_tie += 1
            tie_ds = {}
            for item in sorted_IG[:num_tie]:
                tie_ds[item[0]] = d_num_attributes[item[0]]
            sorted_IG = sorted(tie_ds.items(),key = lambda x: x[1],reverse = True)
            best_d = sorted_IG[0][0]
            best_features,best_labels,branches_count = split_by(best_d)
            self.dim_split = best_d
            self.feature_uniq_split = []
            for feature_value in featuresT[best_d]:
                if feature_value not in self.feature_uniq_split:
                    self.feature_uniq_split.append(feature_value)
            # build child node
            children_sort_info ={}   # { child : attributes }
            for i in range(0,len(best_labels)):
                child_num_cls = len(best_labels[i])
                child_features = np.delete(np.array(best_features[i]),best_d,axis=1).tolist()
                child_labels = best_labels[i]
                child = TreeNode(features = child_features,labels = child_labels,num_cls = child_num_cls)
                if len(child_features) < 1:   # samples run out
                    child.splittable = False
                    child.cls_max = self.cls_max
                if len(child_features[0]) <= 0:
                    child.splittable = False
                else:      # features run out
                    child.split()
                children_sort_info[child] = len(self.feature_uniq_split)
            children_sorted_info = sorted(children_sort_info.items(),key = lambda x: x[1],reverse = True)
            for child_num_attrebutes in children_sorted_info:
                child = child_num_attrebutes[0]
                self.children.append(child)
        else:
            return


        # raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        # print('dim_split ' + str(self.dim_split))
        # print('all children labels')
        # for i in range(0,len(self.children)):
        #     print(self.children[i].labels)
        if self.splittable == False:
            return self.cls_max
        feature_value = feature[self.dim_split]
        idx = self.feature_uniq_split.index(feature_value)
        branch = self.children[idx]
        feature = np.delete(feature,self.dim_split,axis = 0)
        return branch.predict(feature)
        raise NotImplementedError
