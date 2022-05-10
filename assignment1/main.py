import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn import tree
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


class MyDecisionTreeClassifier(DecisionTreeClassifier):

    def __init__(
            self,
            *,
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            ALPHA=0.1
    ):
        self.ALPHA = ALPHA
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def predict_proba(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        arr = self.myPredict(X)
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

    def myPredict(self, x_test):
        arr = []
        for i in range(len(x_test)):
            children_left = self.tree_.children_left
            children_right = self.tree_.children_right
            feature = self.tree_.feature
            threshold = self.tree_.threshold

            stack = [0]
            while stack:
                node_id = stack.pop()
                is_split_node = children_left[node_id] != children_right[node_id]

                if is_split_node:
                    to_test = float(x_test[i, feature[node_id]])
                    rand_num = random.random()
                    if to_test > threshold[node_id]:
                        # should go right but in ALPHA error probability go left
                        if rand_num < self.ALPHA:
                            stack.append(children_left[node_id])
                        # normally go right
                        else:
                            stack.append(children_right[node_id])
                    else:
                        # should go left but in ALPHA error probability go right
                        if rand_num < self.ALPHA:
                            stack.append(children_right[node_id])
                        # normally go left
                        else:
                            stack.append(children_left[node_id])
                else:
                    # its a leaf node
                    arr.append(self.tree_.value[node_id][0])
                    break
        return arr

    def predict(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        arr = self.myPredict(X)
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
        'C:\\Users\\wiggl\\Desktop\\Data_for_UCI_named_1.csv',
        sep=',', header=None, nrows=2000, names=names)

    balance_data = balance_data.drop('stab', axis=1)

    X = balance_data.values[1:, 0:12]
    Y = balance_data.values[1:, 12]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    results = []
    names = []
    cv_results = cross_val_score(MyDecisionTreeClassifier(criterion='entropy', max_depth=7), X_train, Y_train,
                                 cv=RepeatedKFold(n_splits=10, n_repeats=5), scoring='accuracy')
    results.append(cv_results)
    names.append('MODIFIED_DecisionTree')
    msg = "%s: %f" % ('MODIFIED_DecisionTree average score', cv_results.mean())
    print(msg)

    cv_results = cross_val_score(DecisionTreeClassifier(criterion='entropy', max_depth=7), X_train, Y_train,
                                 cv=RepeatedKFold(n_splits=10, n_repeats=5), scoring='accuracy')
    results.append(cv_results)
    names.append('ORIGINAL_DecisionTree')
    msg = "%s: %f" % ('ORIGINAL_DecisionTree average score', cv_results.mean())
    print(msg)

    dtc = MyDecisionTreeClassifier()
    dtc.fit(X_train, Y_train)

    print('Our Classifier metrics:\n')
    # 1. confusion matrix metric
    predictions = dtc.predict(X_test)
    matrix = confusion_matrix(Y_test, predictions)
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])
    print('Confusion Matrix Score: ', accuracy)

    # 2. balanced metric
    print('Balanced Score: ', balanced_accuracy_score(Y_test, predictions))

    # 3. Accuracy metric
    print('Accuracy Score: ', accuracy_score(Y_test, predictions))

    # 4. F1 metric
    print('F1 Score:\n', classification_report(Y_test, predictions))

    dtc_original = DecisionTreeClassifier()
    dtc_original.fit(X_train, Y_train)

    print('Original Classifier metrics:\n')
    # 1. confusion matrix metric
    predictions = dtc_original.predict(X_test)
    matrix = confusion_matrix(Y_test, predictions)
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])
    print('Confusion Matrix Score: ', accuracy)

    # 2. balanced metric
    print('Balanced Score: ', balanced_accuracy_score(Y_test, predictions))

    # 3. Accuracy metric
    print('Accuracy Score: ', accuracy_score(Y_test, predictions))

    # 4. F1 metric
    print('F1 Score:\n', classification_report(Y_test, predictions))

    # Comparing the original classifier to our classification as a plot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Showing up the hole tree
    my_clf = MyDecisionTreeClassifier(criterion='entropy', max_depth=3)
    my_clf.fit(X_train, Y_train)
    tree.plot_tree(my_clf, fontsize=10)
    plt.show()

    models = [(0.1, MyDecisionTreeClassifier(ALPHA=0.1, criterion='entropy', max_depth=3)),
             (0.2, MyDecisionTreeClassifier(ALPHA=0.2, criterion='entropy', max_depth=3)),
             (0.3, MyDecisionTreeClassifier(ALPHA=0.3, criterion='entropy', max_depth=3)),
             (0.4, MyDecisionTreeClassifier(ALPHA=0.4, criterion='entropy', max_depth=3)),
             (0.5, MyDecisionTreeClassifier(ALPHA=0.5, criterion='entropy', max_depth=3)),
             (0.6, MyDecisionTreeClassifier(ALPHA=0.6, criterion='entropy', max_depth=3))]

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for alpha, model in models:
        model.fit(X_train, Y_train)
        score = accuracy_score(Y_test, model.predict(X_test))
        print('With alpha ' + str(alpha), score)
