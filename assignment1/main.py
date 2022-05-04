import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as tc, DecisionTreeClassifier
from sklearn import tree
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold


class MyDecisionTreeClassifier(tc):
    def predict_proba(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        # proba = self.tree_.predict(X)
        proba = self.myPredict(X)
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

    def myPredict(self, X):

        feature = self.tree_.feature
        threshold = self.tree_.threshold

        node_indicator = self.decision_path(X)
        leaf_id = self.myApply(X)

        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]
        print("Rules used to predict sample {id}:\n".format(id=sample_id))
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print(
                "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                "{inequality} {threshold})".format(
                    node=node_id,
                    sample=sample_id,
                    feature=feature[node_id],
                    value=X_test[sample_id, feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id],
                )
            )

        return None

    def myApply(self, X):
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        pass


if __name__ == '__main__':
    balance_data = pd.read_csv(
        'C:\\Users\\wiggl\\Desktop\\Networking\\Data_for_UCI_named.csv',
        sep=',', header=None)

    X = balance_data.values[1:, 0:12]
    Y = balance_data.values[1:, 12]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=7)

    # clf = MyDecisionTreeClassifier(criterion='entropy', max_depth=3)
    clf = tc(criterion='entropy', max_depth=3)
    clf.fit(X_train, y_train)

    k = clf.predict(X_test)

    x = clf.predict_proba(X_test)

    z = accuracy_score(y_test, k)

    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('SVM', SVC(gamma='auto')))
    #models.append(('NB', GaussianNB()))
    models.append(('CART', DecisionTreeClassifier()))
    results = []
    names = []
    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(accuracy_score(y_test, predictions))

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

'''
fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
'''

'''

    print(clf.tree_.feature)

    print(clf.tree_.threshold)

    tree.plot_tree(clf, fontsize=10)
    plt.show()
    print(z)

'''