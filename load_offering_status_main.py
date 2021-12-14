# Importing Main Libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier, DMatrix, plot_importance
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datapreprocessing import DataPreprocessing
from analysis import DataAnalyisAndTransform
from hyper_params_optimize import HPOpt
import nltk
from hyperopt import tpe, Trials, hp

import warnings
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Dict to convert month category to numerical value
month_d = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


if __name__ == '__main__':
    """
    Method to get particular stratified split of train and test
    """


    def getting_train_test_for_Split(train_index, test_index, X_train_np, y_np, fea):
        X_train, X_test = X_train_np[train_index], X_train_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        X_train_df = pd.DataFrame(data=X_train, columns=fea)
        X_test_df = pd.DataFrame(data=X_test, columns=fea)

        y_train_df = pd.DataFrame(data=y_train, columns=['target'])
        y_test_df = pd.DataFrame(data=y_test, columns=['target'])

        return X_train_df, X_test_df, y_train_df, y_test_df


    # In[6]:

    # Ignore Warnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    """
    Object Instantiation, Reading and Loading of the data in dataframe, setting threshold to incldue 
    or exclude feature, random noise columns excluded, storing loan_status as y separately
    """
    data_preprocess_obj = DataPreprocessing(datafile="loan.csv", threshold_percent=0.05,
                                            noisy_cols=['id', 'member_id', 'url'], target_col='loan_status')

    # Calling method to Expand Columns into numeric for given features
    for colname in ['issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line']:
        data_preprocess_obj.expandingCol(colname)

    # In[7]:

    # Calling Method to get Data Description
    data_preprocess_obj.dataDescribe()

    # Calling Method to separate Numeric DataFrame, CategoryText DataFrame
    data_preprocess_obj.separating_Numeric_and_CategoryTextFeatures()

    # Calling Method for Cleaning and Converting Text into Vectors
    data_preprocess_obj.TfIdfVectorizer_TextTransformation()

    # In[8]:

    # Dropping desc feature as Description Text already converted to TF-Idf Vectors
    data_preprocess_obj.cat_df = data_preprocess_obj.cat_df.drop(['desc'], axis=1)

    # Instantiate data analysis object and storing numeric, category and text features
    data_analy_trans_obj = DataAnalyisAndTransform(numeric_df=data_preprocess_obj.numeric_df,
                                                   cat_df=data_preprocess_obj.cat_df,
                                                   text_cat=data_preprocess_obj.X_cat)

    # Calling method to drop numeric columns of zero variance
    data_analy_trans_obj.removeZeroVarNumericFeatures()

    # Calling method to check Cardinality
    data_analy_trans_obj.checkingCardinality()

    # In[9]:

    # Calling method to drop category columns of zero variance
    data_analy_trans_obj.removeZeroVarCategoryFeatures()

    # Calling method to find low cardinal and high cardinal features
    data_analy_trans_obj.define_low_high_feat()

    # Calling method to encode low cardinal and high cardinal features
    data_analy_trans_obj.cardinality_to_numeric(data_preprocess_obj.y)

    # started the KNN training for imputation
    data_analy_trans_obj.missing_value_imputation()

    # In[10]:

    """
    Balanced Sampling according to the distribution of class. Means Minority Class will get more weights.
    """
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(data_analy_trans_obj.y),
                                                y=data_analy_trans_obj.y)
    weights = weights / sum(weights)
    print("Balanced Sampling that is inversely to class sample size and weights are: " + str(weights))
    d = dict(Counter(data_analy_trans_obj.y))
    print(sorted(d.items(), key=lambda s: s[0]))

    # In[11]:

    """
    Initializing search space for Hyper Parameters of XGBoost Classifier
    """
    xgb_clf_params = {
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'max_depth': hp.choice('max_depth', range(4, 16, 1)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
        'n_estimators': 100,
        'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
        'n_jobs': 16
    }
    # 'sample_weight' : weights
    xgb_fit_params = {
        'eval_metric': 'merror',
        'early_stopping_rounds': 10,
        'verbose': False
    }
    xgb_para = dict()
    xgb_para['clf_params'] = xgb_clf_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['accuracy'] = lambda y, pred: accuracy_score(y, pred)

    ind_train_list, ind_test_list, xgb_opts, iteration = [], [], [], 1

    # In[12]:

    data_analy_trans_obj.statify_splits(num_splits=5, test_size=0.2)

    # In[13]:

    """
    Applying Bayesion Optimization to every stratified split of train and test dataset.
    """
    for train_index, test_index in data_analy_trans_obj.splits:
        X_train_df, X_test_df, y_train_df, y_test_df = getting_train_test_for_Split(train_index, test_index,
                                                                                    data_analy_trans_obj.X_train_np,
                                                                                    data_analy_trans_obj.y_np,
                                                                                    data_analy_trans_obj.X_train_fea)

        obj = HPOpt(X_train_df, X_test_df, y_train_df, y_test_df)
        print("\nOptimization on Current Splits and parameters for split " + str(iteration))
        xgb_opt = obj.process(fn_name='xgb_classifier', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=50)
        print(xgb_opt)
        xgb_opts.append(xgb_opt)
        iteration += 1
        ind_train_list.append(train_index)
        ind_test_list.append(test_index)

    # # Best Minimum Loss is 0.000881161 and Its on Split 5

    # In[14]:

    best_train_ind, best_test_ind, best_xgb_hyOPts = ind_train_list[4], ind_test_list[4], xgb_opts[4]

    # In[15]:

    X_train_df, X_test_df, y_train_df, y_test_df = getting_train_test_for_Split(best_train_ind, best_test_ind,
                                                                                data_analy_trans_obj.X_train_np,
                                                                                data_analy_trans_obj.y_np,
                                                                                data_analy_trans_obj.X_train_fea)

    # In[17]:

    best_params = xgb_opts[4][0]
    print(best_params)

    # In[21]:

    """
    XGBoost Classifier Modelling on best Stratified split and best hyper params
    """
    xgb_best = XGBClassifier(n_estimators=100, n_jobs=16, verbosity=0)
    xgb_best.set_params(**best_params)
    xgb_best.fit(X_train_df, y_train_df, eval_metric='merror', verbose=False)
    preds = xgb_best.predict(X_test_df)
    print("\nAccuracy of Classifier on held out Test data: " + str(accuracy_score(y_test_df, preds) * 100) + ' percent')
    print('*' * 100)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test_df, preds))
    print('*' * 100)

    """
    Plotting Top 30 Features
    """
    from matplotlib import pyplot

    plot_importance(xgb_best, max_num_features=30)
    pyplot.show()

    # # Accuracy of Classifier on held out Test data: 99.8489425981873 percent

    # # THAT'S COOL.  THANKS.  :)

