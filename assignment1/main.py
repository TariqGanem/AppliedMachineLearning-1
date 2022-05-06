import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn import tree
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    ALPHA_PROB = 0.1

    def predict_proba(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        arr = []
        for i in range(len(X)):
            arr.append(self.myPredict(X, i))
        proba = np.array(arr)
        if self.n_outputs_ == 1:
            proba = proba[:, : self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

        return all_proba

    def myPredict(self, x_test, sample_id):

        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()
            if children_left[node_id] == children_right[node_id]:
                return self.tree_.value[node_id][0]

            to_test = float(x_test[sample_id, feature[node_id]])
            if to_test <= threshold[node_id]:
                if random.random() >= self.ALPHA_PROB:
                    stack.append(children_left[node_id])
                else:
                    stack.append(children_right[node_id])
            else:
                if random.random() >= self.ALPHA_PROB:
                    stack.append(children_right[node_id])
                else:
                    stack.append(children_left[node_id])

    def predict(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        arr = []
        for i in range(len(X)):
            arr.append(self.myPredict(X, i))
        proba = arr
        proba = np.array(proba)
        n_samples = X.shape[0]

        # Classification
        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)

            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]


if __name__ == '__main__':
    names = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4', 'stab', 'stabf']
    balance_data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv',
        sep=',', header=None, nrows=2000, names=names)

    balance_data = balance_data.drop('stab', axis=1)

    X = balance_data.values[1:, 0:12]
    Y = balance_data.values[1:, 12]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=None)

    my_clf = MyDecisionTreeClassifier(criterion='entropy', max_depth=3)

    my_clf.fit(X_train, y_train)

    models = [('MODIFIED', MyDecisionTreeClassifier()), ('ORIGINAL', DecisionTreeClassifier())]
    results = []
    names = []
    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    knn = MyDecisionTreeClassifier()
    knn.fit(X_train, y_train)

    # 1. confusion matrix metric
    predictions = knn.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])
    print('Confusion Matrix Score: ', accuracy)

    # 2. balanced metric
    print('Balanced Score: ', balanced_accuracy_score(y_test, predictions))

    # 3. Accuracy metric
    print('Accuracy Score:\n', accuracy_score(y_test, predictions))

    # 4. F1 metric
    print('F1 Score:\n', classification_report(y_test, predictions))

    # Comparing the original classifier to our classification as a plot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Showing up the hole tree
    tree.plot_tree(my_clf, fontsize=10)
    plt.show()

