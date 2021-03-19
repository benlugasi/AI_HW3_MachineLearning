from ID3 import ID3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class CostSensitiveID3(ID3):
    def fit(self, X, y, m=2, k=7):
        c = self.major_class(y)
        self._feature_list = X.keys()
        self._tree = self.generate_tree(X, y, c, m, k)

    def generate_tree(self, X, y, default, m, k=7):
        if len(X) < m:
            return None, [], default
        c = self.major_class(y)
        B_cnt, M_cnt = self.countBM(y)
        if self.is_unique(y) or B_cnt*k < M_cnt:
            return None, [], c
        feature_selected = self.Max_IG(X, y)[:-1]  # a list (feature: str, separator: float)
        l_indexes = X[feature_selected[0]] < feature_selected[1]
        g_indexes = X[feature_selected[0]] >= feature_selected[1]
        sub_trees = [
            (self.generate_tree(X.loc[l_indexes], y.loc[l_indexes], c, m, k)),
            (self.generate_tree(X.loc[g_indexes], y.loc[g_indexes], c, m, k))]
        return feature_selected, sub_trees, c

    @staticmethod
    def countBM(y):
        y = y.to_numpy()
        B_cnt = len(y[np.where(y == 'B')])
        M_cnt = len(y[np.where(y == 'M')])
        return B_cnt, M_cnt

def ex4_2_experiment_CV(X, y):
    clf = CostSensitiveID3()
    k_list = range(1, 15)
    kf = KFold(n_splits=5, shuffle=True, random_state=205816655)
    loss_val = []
    avg_loss = 0
    for k in k_list:
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            clf.fit(X_train, y_train, 5, k)
            scores.append(clf.loss(X_test, y_test))
            avg_loss = np.average(scores)
        loss_val.append(avg_loss)
        print("k =", k, "loss =", avg_loss)
    plt.title('Ex.4.3')
    plt.ylabel('loss')
    plt.xlabel('k')
    plt.plot(k_list, loss_val)
    plt.show()


def ex4_2(X_train, y_train, X_test, y_test):
    clf = CostSensitiveID3()
    clf.fit(X_train, y_train, 5, 7)
    print(clf.loss(X_test, y_test))


if __name__ == '__main__':
    train = pd.read_csv(r"train.csv")
    X_train = train.drop('diagnosis', axis=1)
    y_train = train['diagnosis']

    test = pd.read_csv(r"test.csv")
    X_test = test.drop('diagnosis', axis=1)
    y_test = test['diagnosis']

    # ex4_2_experiment1(X_train, y_train, X_test, y_test)
    # ex4_2_experiment_CV(X_train, y_train)
    ex4_2(X_train, y_train, X_test, y_test)
