from KNNForest import KNNForest
from ID3 import ID3
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class ImprovedKNNForest(KNNForest):
    def __init__(self, K, N, s):
        super().__init__(K, N, 0.6)
        self.p_list = [0.4, 0.6, 0.7]
        self.s = s
        self.classifiers = []

    def fit(self, X, y):
        self.train_max = X.max()
        self.train_min = X.min()
        X = self.minmax_norm_X(X)
        self.classifiers = np.unique(y)
        for i in range(self.N):
            sub_X = X.sample(frac=self.p_list[i % len(self.p_list)], random_state=i)
            centroid = [np.average(sub_X[feature].to_numpy()) for feature in sub_X.keys()]
            self.centroid_list.append(centroid)
            sub_y = y[sub_X.index]
            self.id3_list.append(ID3())
            self.id3_list[i].fit(sub_X, sub_y, 5)

    def predict(self, X):
        X = self.minmax_norm_X(X)
        y_ans_matrix = [id3.predict(X) for id3 in self.id3_list]
        y_ans = []
        for j, (index, o) in enumerate(X.iterrows()):
            K_closest = self.K_closest_nb(o)
            board_answers = [(self.get_most_frequent_ans(y_ans_matrix[i[0]][j]), i[1]) for i in K_closest]
            y_ans.append(self.get_best_c_rank(board_answers))
        return y_ans

    def K_closest_nb(self, o):
        if self.K == self.N:
            return range(self.N)
        distance_from_o = [self.dist_between_examples(o, centroid) for centroid in self.centroid_list]
        indexes = np.argpartition(distance_from_o, self.K)[:self.K]
        pruned_indexes = [i for i in indexes if distance_from_o[i] <= float('inf')]  # prune
        if len(pruned_indexes) == 0:
            return [(i, distance_from_o[i]) for i in indexes]
        return [(i, distance_from_o[i]) for i in pruned_indexes]

    def get_best_c_rank(self, board_answers):
        dist_sum = sum(member[1] for member in board_answers)
        c_ranks = np.zeros(len(self.classifiers))
        for member in board_answers:
            c_ranks[np.where(self.classifiers == member[0])] += member[1]/dist_sum
        return self.classifiers[int(np.argmax(c_ranks))]

    def minmax_norm_X(self, X):
        return (X - self.train_min) / (self.train_max - self.train_min)


def ex7_2(X_train, y_train, X_test, y_test):
    clf = ImprovedKNNForest(8, 9, 0.5)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def find_best_s(X, y):
    s_list = np.arange(0.5, 5, 0.25)
    kf = KFold(n_splits=5, shuffle=True, random_state=205816655)
    final_score = []
    max_s = 0.5
    max_score = float('-inf')
    for s in s_list:
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            clf = ImprovedKNNForest(8, 9, s)
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))
        curr_score = np.average(scores)
        print(s, curr_score)
        final_score.append(curr_score)
        if curr_score > max_score:
            max_s = s
            max_score = curr_score
    print("max:", max_s)
    plt.title('Ex.7.2')
    plt.ylabel('score')
    plt.xlabel('s')
    plt.tick_params('y')
    plt.plot(s_list, final_score)
    plt.show()


if __name__ == '__main__':
    train = pd.read_csv(r"train.csv")
    X_train = train.drop('diagnosis', axis=1)
    y_train = train['diagnosis']

    test = pd.read_csv(r"test.csv")
    X_test = test.drop('diagnosis', axis=1)
    y_test = test['diagnosis']
    ex7_2(X_train, y_train, X_test, y_test)
    # find_best_s(X_train, y_train)
