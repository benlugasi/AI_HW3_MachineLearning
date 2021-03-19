from sklearn import base
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ID3(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self):
        self._feature_list = None
        self._tree = None

    def fit(self, X, y, m=2):
        c = self.major_class(y)
        self._feature_list = X.keys()
        self._tree = self.generate_tree(X, y, c, m)

    def loss(self, X, y):
        y = y.to_numpy()
        y_ans = self.predict(X)
        n = len(y)
        loss_val = 0
        for i in range(n):
            if y_ans[i] == 'B' and y[i] == 'M':  # FN
                loss_val += 1
            elif y_ans[i] == 'M' and y[i] == 'B':  # FP
                loss_val += 0.1
        loss_val = loss_val / n
        return loss_val
    
    def predict(self, X):
        y_ans = []
        for index, row in X.iterrows():
            y_ans.append(self.classify(row, self._tree))
        return y_ans

    def classify(self, o, tree):
        feature, subtree, c = tree
        if not subtree:
            return c
        if o[feature[0]] < feature[1]:
            return self.classify(o, subtree[0])
        else:
            return self.classify(o, subtree[1])

    def generate_tree(self, X, y, default, m):
        if len(X) < m:
            return None, [], default
        c = self.major_class(y)
        if self.is_unique(y):
            return None, [], c
        feature_selected = self.Max_IG(X, y)[:-1]  # a list (feature: str, separator: float)
        l_indexes = X[feature_selected[0]] < feature_selected[1]  # {x in X | f(x) < sep}
        g_indexes = X[feature_selected[0]] >= feature_selected[1]  # {x in X | f(x) >= sep}
        sub_trees = [
            (self.generate_tree(X.loc[l_indexes], y.loc[l_indexes], c, m)),
            (self.generate_tree(X.loc[g_indexes], y.loc[g_indexes], c, m))]
        return feature_selected, sub_trees, c

    def Max_IG(self, X, y):
        IG_list = []
        for i, feature in enumerate(self._feature_list):
            e = [feature]
            e.extend(self.IG(feature, X, y))
            e.extend([i])
            IG_list.append(e)
        return max(IG_list, key=lambda x: (x[2], x[3]))

    def IG(self, feature: str, X, y):
        feature_examples = X[feature].to_numpy()
        y = y.to_numpy()
        f_range = self.get_range(feature_examples)
        total_entropy = self.entropy(y)
        sep_list = []
        for separator in f_range:
            less_e = y[np.where(feature_examples <= separator)]
            less_e_size = len(less_e) / len(y)
            greater_e = y[np.where(feature_examples > separator)]
            greater_e_size = len(greater_e) / len(y)
            entropy_val = self.entropy(less_e) * less_e_size + self.entropy(greater_e) * greater_e_size
            sep_list.append((separator, total_entropy - entropy_val))
        return max(sep_list, key=lambda x: x[1])

    @staticmethod
    def major_class(y):
        y = y.to_numpy()
        unique, pos = np.unique(y, return_inverse=True)
        max_pos = np.argmax(np.bincount(pos))
        return unique[max_pos]

    @staticmethod
    def is_unique(y):
        y = y.to_numpy()
        return len(np.unique(y)) == 1

    @staticmethod
    def get_range(X):
        X_copy = np.unique(np.copy(X))
        X_copy.sort()
        return [(X_copy[i] + X_copy[i + 1]) / 2 for i in range(len(X_copy) - 1)]

    @staticmethod
    def entropy(y):
        classifiers = np.unique(y)
        entropy_val = 0
        for c in classifiers:
            c_cnt = np.count_nonzero(y == c)
            relative_c_size = c_cnt / len(y)
            entropy_val += -relative_c_size * np.log2(relative_c_size)
        return entropy_val


def ex1_1(X_train, y_train, X_test, y_test):
    clf = ID3()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def experiment(X: pd.DataFrame, y: pd.DataFrame):
    """takes 2 arguments clf:
        - X(pd.DataFrame) the Training data
        - y(pd.DataFrame) The target variable for supervised learning problems
        example:
        train = pd.read_csv(r"data\train.csv")
        X_train = train.drop('diagnosis', axis=1)
        y_train = train['diagnosis']
        experiment(X_train, y_train)
    """
    clf = ID3()
    kf = KFold(n_splits=5, shuffle=True, random_state=205816655)
    final_score = []
    m_list = [5, 10, 15, 20, 25]
    for m in m_list:
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            clf.fit(X_train, y_train, m)
            scores.append(clf.score(X_test, y_test))
        final_score.append(np.average(scores))
    print(final_score)
    plt.title('Ex.3.3')
    plt.ylabel('score')
    plt.xlabel('m')
    plt.tick_params('y')
    plt.plot(m_list, final_score)
    plt.show()


def ex3_4(X_train, y_train, X_test, y_test):
    clf = ID3()
    clf.fit(X_train, y_train, 5)
    print(clf.score(X_test, y_test))


def ex4_1(X_train, y_train, X_test, y_test):
    clf = ID3()
    clf.fit(X_train, y_train, 5)
    print(clf.loss(X_test, y_test))


if __name__ == '__main__':
    train = pd.read_csv(r"train.csv")
    X_train = train.drop('diagnosis', axis=1)
    y_train = train['diagnosis']

    test = pd.read_csv(r"test.csv")
    X_test = test.drop('diagnosis', axis=1)
    y_test = test['diagnosis']

    ex1_1(X_train, y_train, X_test, y_test)
    # experiment(X_train, y_train)
    # ex3_4(X_train, y_train, X_test, y_test)
    # ex4_1(X_train, y_train, X_test, y_test)
