from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp
from xgboost import XGBClassifier, DMatrix, plot_importance
from sklearn.ensemble import RandomForestClassifier


class HPOpt(object):
    """
    Bayesion Optimization using HyperOpt and XGBoostClassifier Model was used.
    """

    def __init__(self, x_train, x_test, y_train, y_test, weights=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.weights = weights

    """
    Finding Best hyper-parameters for model from Search space and by using Tree Parser Estimator as 
    selection method to find next set of hyperparameters.
    """

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        return result, trials

    """
    Setting params for XGBoostClassifier.
    """

    def xgb_classifier(self, para):
        classifier = XGBClassifier(**para['clf_params'])
        return self.train_classifier(classifier, para)

    """
    Training of XGBoostClassifier.
    """

    def train_classifier(self, classifier, para):
        classifier.fit(self.x_train, self.y_train, sample_weight=self.weights,
                       eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                       **para['fit_params'])
        pred = classifier.predict(self.x_test)
        accuracy = para['accuracy'](self.y_test, pred)
        return {'loss': 1 - accuracy, 'status': STATUS_OK}
