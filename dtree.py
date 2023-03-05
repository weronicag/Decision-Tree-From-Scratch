import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


def gini(y):
    "Return the gini impurity score for values in y"
    _, counts = np.unique(y, return_counts=True)
    n = len(y)
    return 1 - np.sum((counts / n)**2)


def bestsplit(self, X, y, loss):
    """ Finds the optimal parameters (split feature and value) to optimize minimize cost"""
    y = y.reshape(-1, 1)
    X_y = np.hstack([X, y])
    # choosing k possible split to increase efficiency and generality
    k = 11
    best = {'feature': -1, 'split': -1, 'loss': loss(y)}
    for feature in range(X.shape[1]):
        possible_splits = np.unique(X[:, feature])
        k_selected_split = np.random.choice(possible_splits, k)
        for split in k_selected_split:
            # evaluate y values in the left and right split
            lefty = X_y[X_y[:, feature] <= split][:, -1]
            righty = X_y[X_y[:, feature] > split][:, -1]
            if len(lefty) > self.min_samples_leaf or len(righty) > self.min_samples_leaf:
                weighted_avg_loss = (
                    len(lefty)*loss(lefty) + len(righty)*loss(righty)) / len(y)
                if weighted_avg_loss == 0:
                    return feature, split
                if weighted_avg_loss < best['loss']:
                    best['loss'] = weighted_avg_loss
                    best['feature'] = feature
                    best['split'] = split
    # return the feature and split that gives the bestsplit to reduce varriance/ has the purest y
    return best['feature'], best['split']


class DecisionNode:
    """This class implements a internal decision node in the tree structure"""

    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if self is None:
            return None
        if self is LeafNode:
            return LeafNode.predict(x_test)
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)


class LeafNode:
    """This class implements a leaf node in the tree structure"""

    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def __str__(self):
        return str(self.parent)

    def predict(self, x_test):
        return self.prediction


class DecisionTree621:
    """This class implements the basic tree structure for both a regression and classifier tree"""

    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.
        """
        y = y.reshape(-1, 1)
        X_y = np.hstack([X, y])

        # return a leaf node if reached min_samples_leaf, purest possible y, or no more split values
        if len(y) <= self.min_samples_leaf or len(np.unique(y)) == 1 or np.unique(X, axis=0).shape[0] == 1:
            return self.create_leaf(y)
        feature, split = bestsplit(self, X, y, self.loss)
        # if no better split
        if feature == -1:
            return self.create_leaf(y)

        lchild = self.fit_(X_y[X_y[:, feature] <= split]
                           [:, :-1], X_y[X_y[:, feature] <= split][:, -1])
        rchild = self.fit_(X_y[X_y[:, feature] > split]
                           [:, :-1], X_y[X_y[:, feature] > split][:, -1])
        return DecisionNode(feature, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return self.root.predict(X_test)


class RegressionTree621(DecisionTree621):
    """This class implements a Regression Tree comparable to Scikit-Learn's"""

    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_predict = [self.predict(x_test) for x_test in X_test]
        r2 = r2_score(y_test, y_predict)
        return r2

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    """This class implements a Classifier Tree comparable to Scikit-Learn's"""

    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_predict = [self.predict(x_test) for x_test in X_test]
        accuracy_score_result = accuracy_score(y_test, y_predict)
        return accuracy_score_result

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        y_values = y.reshape(y.shape[0],)
        # creates a LeafNode with mode(y)
        return LeafNode(y, np.bincount(y_values.astype(int)).argmax())
   
