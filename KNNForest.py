from ID3 import ID3
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import base
import pandas as pd
import numpy as np


class KNNForest(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, K, N, p):
        self.id3_list = []
        self.centroid_list = []
        self.N = N
        self.K = K
        self.p = p

    def fit(self, X, y):
        for i in range(self.N):
            sub_X = X.sample(frac=self.p, random_state=i)
            centroid = [np.average(sub_X[feature].to_numpy()) for feature in sub_X.keys()]
            self.centroid_list.append(centroid)
            sub_y = y[sub_X.index]
            self.id3_list.append(ID3())
            self.id3_list[i].fit(sub_X, sub_y, 5)

    def predict(self, X):
        y_ans_matrix = [id3.predict(X) for id3 in self.id3_list]
        y_ans = []
        for j, (index, o) in enumerate(X.iterrows()):
            K_closest = self.K_closest_nb(o)
            board_answers = [self.get_most_frequent_ans(y_ans_matrix[i][j]) for i in K_closest]
            y_ans.append(self.get_most_frequent_ans(board_answers))
        return y_ans

    def K_closest_nb(self, o):
        if self.K == self.N:
            return range(self.N)
        distance_from_o = [self.dist_between_examples(o, centroid) for centroid in self.centroid_list]
        return np.argpartition(distance_from_o, self.K)[:self.K]

    @staticmethod
    def dist_between_examples(x, y):
        return np.linalg.norm(x - y)

    @staticmethod
    def get_most_frequent_ans(y):
        unique, pos = np.unique(y, return_inverse=True)
        max_pos = np.argmax(np.bincount(pos))
        return unique[max_pos]


def ex6_1(X_train, y_train, X_test, y_test):
    clf = KNNForest(8, 9, 0.6)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def find_best_k_n_p(X, y):
    N_list = range(3, 10)
    kf = KFold(n_splits=5, shuffle=True, random_state=205816655)
    for p in np.arange(0.3, 0.701, 0.1):
        max_k, max_N, max_p = 0, 0, 0
        max_score = float('-inf')
        final_score = []
        n = []
        k = []
        for N in N_list:
            K_list = range(2, N)
            for K in K_list:
                scores = []
                n.append(N)
                k.append(K)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.loc[train_index], X.loc[test_index]
                    y_train, y_test = y.loc[train_index], y.loc[test_index]
                    clf = KNNForest(K, N, p)
                    clf.fit(X_train, y_train)
                    scores.append(clf.score(X_test, y_test))
                curr_score = np.average(scores)
                final_score.append(curr_score)
                if curr_score > max_score:
                    max_k, max_N = K, N
                    max_score = curr_score

        print("max:", max_score, "k", max_k, "p", p, "N", max_N)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.title('Ex.6.1 p =' + str(round(p, 1)))
        plt.ylabel('k')
        plt.tick_params('y')
        plt.xlabel('n')
        img = ax.scatter(n, k, c=final_score, cmap='viridis')
        fig.colorbar(img)
    plt.show()


if __name__ == '__main__':
    train = pd.read_csv(r"train.csv")
    X_train = train.drop('diagnosis', axis=1)
    y_train = train['diagnosis']

    test = pd.read_csv(r"test.csv")
    X_test = test.drop('diagnosis', axis=1)
    y_test = test['diagnosis']
    ex6_1(X_train, y_train, X_test, y_test)
    # find_best_k_n_p(X_train, y_train)
