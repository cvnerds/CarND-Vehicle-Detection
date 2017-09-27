from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import tree
import time


class Classifier:
    def __init__(self, model='linear_svm'):
        if model == 'decision_tree':
            self.classifier = tree.DecisionTreeClassifier()
        elif model == 'linear_svm':
            self.classifier = LinearSVC()
        else:
            self.classifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
            #kernel = 'poly'
            #kernel = 'sigmoid'
    
    def train(self, X, y):
        t = time.time()
        self.classifier.fit(X, y)
        t2 = time.time()
        return self, round(t2-t, 2)

    def score(self, X, y):
        return self.classifier.score(X,y)

    def predict(self, X):
        return self.classifier.predict(X)
