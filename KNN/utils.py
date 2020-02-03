import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    label_length=len(real_labels)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(0,label_length):
        if real_labels[i] == 1 and predicted_labels[i] == 1:
            TP += 1
        elif real_labels[i] == 1 and predicted_labels[i] == 0:
            FP += 1
        elif real_labels[i] == 0 and predicted_labels[i] == 0:
            TN += 1
        elif real_labels[i] == 0 and predicted_labels[i] == 1:
            FN += 1
    F1_score = 2 * TP / (2 * TP + FN + FP)
    return float(F1_score)
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        D = len(point1)
        sum = 0
        for i in range(0,D):
            sum += (abs(point1[i] - point2[i])) ** 3
        minkowski_distance = sum ** (1/3)
        return float(minkowski_distance)
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        D = len(point1)
        sum = 0
        for i in range(0,D):
            sum += (point1[i] - point2[i]) ** 2
        euclidean_distance = sum ** (1/2)
        return float(euclidean_distance)
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        D = len(point1)
        inner_product_distance = 0
        for i in range(0,D):
            inner_product_distance += point1[i] * point2[i]
        return float(inner_product_distance)
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        D = len(point1)
        point1_length = 0.0
        point2_length = 0.0
        inner_product = 0.0
        for i in range(0,D):
            inner_product += point1[i] * point2[i]
            point1_length += point1[i] ** 2
            point2_length += point2[i] ** 2
        point1_length = np.sqrt(point1_length)
        point2_length = np.sqrt(point2_length)
        cosine_similarity_distance =1 - inner_product / (point1_length * point2_length)
        return float(cosine_similarity_distance)
        raise NotImplementedError


    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        D = len(point1)
        sum = 0
        for i in range(0,D):
            sum += (point1[i] - point2[i]) ** 2
        sum = (-1) * sum / 2
        gaussian_kernel_distance = -np.exp(sum)
        return float(gaussian_kernel_distance)
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        tuning_info = {}   #  {model_info : F1_score}
        for function_name,function in distance_funcs.items():
            for k in range(1,min(30,len(x_train)),2):
                model_info = (function_name,k)
                model = KNN(k,function)
                model.train(x_train,y_train)
                y_predict = model.predict(x_val)
                F1_score = f1_score(real_labels=y_val, predicted_labels=y_predict)
                tuning_info[model_info] = F1_score
        sorted_tuning_info = sorted(tuning_info.items(),key=lambda x : x[1],reverse = True)   #List [(model_info,F1_score)]
        num_tie_models=1   #num_tie_models = 1 means have only best model
        for i in range(0,len(sorted_tuning_info)-1):
            if sorted_tuning_info[i][1] != sorted_tuning_info[i+1][1]:
                break
            num_tie_models += 1
        print(num_tie_models)
        best_models = sorted_tuning_info[:num_tie_models]
        func_scoring_dic = {
            'euclidean': 40,
            'minkowski': 80,
            'gaussian': 120,
            'inner_prod': 160,
            'cosine_dist': 200,
        }
        scores = []
        for model in best_models:
            scores.append(func_scoring_dic[model[0][0]] + model[0][1])
        best_score_index = np.argmin(scores)
        best_model = best_models[best_score_index]

        # You need to assign the final values to these variables
        self.best_k = best_model[0][1]
        self.best_distance_function = best_model[0][0]
        self.best_model = KNN(k=self.best_k,distance_function=distance_funcs[self.best_distance_function])
        self.best_model.train(x_train,y_train)
        return
        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        tuning_info = {}   #  {model_info : F1_score}
        for scaler_name,scaler_type in scaling_classes.items():
            for function_name,function in distance_funcs.items():
                for k in range(1,min(30,len(x_train)),2):
                    model_info = (function_name,k,scaler_name)
                    model = KNN(k,function)
                    scaler = scaler_type()
                    x_train_scaled = scaler(x_train)
                    x_val_scaled = scaler(x_val)
                    model.train(x_train_scaled,y_train)
                    y_predict = model.predict(x_val_scaled)
                    F1_score = f1_score(real_labels=y_val, predicted_labels=y_predict)
                    tuning_info[model_info] = F1_score
        sorted_tuning_info = sorted(tuning_info.items(),key=lambda x : x[1],reverse = True)   #List [(model_info,F1_score)]
        num_tie_models=1   #num_tie_models = 1 means have only best model
        for i in range(0,len(sorted_tuning_info)-1):
            if sorted_tuning_info[i][1] != sorted_tuning_info[i+1][1]:
                break
            num_tie_models += 1
        best_models = sorted_tuning_info[:num_tie_models]
        func_scoring_dic = {
            'euclidean': 40,
            'minkowski': 80,
            'gaussian': 120,
            'inner_prod': 160,
            'cosine_dist': 200,
        }
        scaler_scoring_dic = {
            'min_max_scale': 0,
            'normalize': 300,
        }
        scores = []
        for model in best_models:
            scores.append(func_scoring_dic[model[0][0]] + scaler_scoring_dic[model[0][2]] + model[0][1])
        for i in range(0,len(best_models)):
            print(best_models[i])
            print(scores[i])
        best_score_index = np.argmin(scores)
        best_info = best_models[best_score_index]
        # You need to assign the final values to these variables
        self.best_k = best_info[0][1]
        self.best_distance_function = best_info[0][0]
        self.best_scaler = best_info[0][2]
        self.best_model = KNN(k=self.best_k,distance_function=distance_funcs[self.best_distance_function])
        best_scaler_type = scaling_classes[self.best_scaler]()
        x_train = best_scaler_type(x_train)
        self.best_model.train(x_train,y_train)
        return
        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
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

        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_check = True
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if self.first_check == True:
            self.minmax = []
            featuresT=np.array(features).T
            for ith_feature in featuresT:
                self.minmax.append([min(ith_feature),max(ith_feature)])
            self.first_check = False
            new_features = []
            for i in range(0,len(featuresT)):
                new_feature = []
                for ith_num in featuresT[i]:
                    if self.minmax[i][1] == self.minmax[i][0]:
                        new_num = 0
                    else:
                        new_num = (ith_num - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
                    new_feature.append(new_num)
                new_features.append(new_feature)
            return(np.array(new_features).T.tolist())
        else:
            featuresT = np.array(features).T
            new_features = []
            for i in range(0,len(featuresT)):
                new_feature = []
                for ith_num in featuresT[i]:
                    if self.minmax[i][1] == self.minmax[i][0]:
                        new_num = 0
                    else:
                        new_num = (ith_num -self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0])
                    new_feature.append(new_num)
                new_features.append(new_feature)
            return(np.array(new_features).T.tolist())
        raise NotImplementedError
