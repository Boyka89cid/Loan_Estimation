# Importing Main Libraries
import pandas as pd
import category_encoders as ce
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score


class DataAnalyisAndTransform(object):
    """
    Loading Numeric Data, category Data and Text Data into respective Pandas DataFrame.
    """

    def __init__(self, numeric_df=None, cat_df=None, text_cat=None):
        super(DataAnalyisAndTransform, self).__init__()
        self.numeric_df = numeric_df
        self.cat_df = cat_df
        self.text_df = text_cat

    """
    Method to Remove numeric features having zero Variance
    """

    def removeZeroVarNumericFeatures(self):
        print("\nRemoving features of Zero Variance and shape of Numeric Data Frame: ")
        self.numeric_df = self.numeric_df.loc[:, self.numeric_df.std() > 0.0]
        print(self.numeric_df.shape)
        self.numeric_feat = list(self.numeric_df.columns.values)

    """
    Method to check cardinality of categorical features
    """

    def checkingCardinality(self):
        print("\nChecking Cardinalty of Features before encoding category into numeic Values: ")
        self.cardinality_df = pd.DataFrame(columns=['Feature', 'Cardinality'])
        self.cardinality_df['Feature'] = list(self.cat_df.columns.values)
        self.cardinality_df['Cardinality'] = list(self.cat_df.nunique().values)
        print(self.cardinality_df)

    """
    Method to Remove categorical features having zero Variance .i.e. their cardinality is 1.
    """

    def removeZeroVarCategoryFeatures(self):
        zero_var_cat_fea = self.cardinality_df[self.cardinality_df['Cardinality'] == 1]
        zero_var_cat_fea_list = list(zero_var_cat_fea['Feature'].values)
        print("\nPrinting Features with zero variance and dropping from category dataFrame: ")
        print(zero_var_cat_fea_list)
        self.cat_df = self.cat_df.drop(zero_var_cat_fea_list, axis=1)

        for c in zero_var_cat_fea_list:
            self.cardinality_df = self.cardinality_df[self.cardinality_df.Feature != c]
        self.cardinality_df = self.cardinality_df.reset_index(drop=True)
        print("\nRemoved Zero Variance Features From Cardinality DataFrame: ")
        print(self.cardinality_df)

    """
    Method to find low and high cardinal categorical features .i.e. threshold is 50.
    """

    def define_low_high_feat(self):
        high_cardinal_fea_df = self.cardinality_df[self.cardinality_df['Cardinality'] >= 50]
        self.high_cardinal_features = list(high_cardinal_fea_df['Feature'].values)
        print("\nHigh Cardinal Features: ")
        print(self.high_cardinal_features)

        low_cardinal_fea_df = self.cardinality_df[self.cardinality_df['Cardinality'] < 50]
        self.low_cardinal_features = list(low_cardinal_fea_df['Feature'].values)
        print("\nLow Cardinal Features: ")
        print(self.low_cardinal_features)

    """
    Method to convert low and high cardinal categorical features into numeric values. Cardinality 
    threshold is 50. For Low cardinality, Binary Encoder is used. It implies max 6 columns will be used
    to encode cardinality of 50 unique features  as Base 2 -log(50) rounds to 6.
    For High cardinality, LeaveOneOutEncoder is used.
    """

    def cardinality_to_numeric(self, y):
        encoder_High_Cardinal = ce.LeaveOneOutEncoder(drop_invariant=True, handle_missing='return_nan', random_state=2)
        encoder_Low_Cardinal = ce.BinaryEncoder(drop_invariant=True, handle_missing='return_nan')
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(y)

        self.high_cardinal_clean = encoder_High_Cardinal.fit_transform(
            self.cat_df[self.high_cardinal_features], self.y_encoded)
        self.low_cardinal_clean = encoder_Low_Cardinal.fit_transform(
            self.cat_df[self.low_cardinal_features], self.y_encoded)
        print("\nDimensionality after Leave One Out and Binary Encoding: ")
        print(self.high_cardinal_clean.shape)
        print(self.low_cardinal_clean.shape)

    """
    Method to impute missing values using K-NN nearest neighbour technique. It implies missing value are
    imputed by weighted mean of k nearest neighbour according to their distances.
    """

    def missing_value_imputation(self):
        # started the KNN training for imputation
        self.categorical_feat = list(self.high_cardinal_clean.columns.values) + list(
            self.low_cardinal_clean.columns.values)
        y_df = pd.DataFrame(data=self.y_encoded, columns=["y_target"])
        missing_df = pd.concat([self.numeric_df, self.high_cardinal_clean, self.low_cardinal_clean, y_df], axis=1)
        imputed_training = fast_knn(missing_df.values, k=30)

        self.imputed_df = pd.DataFrame(data=imputed_training, columns=list(missing_df.columns.values))
        print("After Imputation Number of Missing Values for each Feature: ")
        print(self.imputed_df.isnull().sum().values)

        self.y = self.imputed_df['y_target']
        self.imputed_df = self.imputed_df.drop(['y_target'], axis=1)
        feats = ['feat_' + str(i) for i in range(2000)]
        TfIdf_df = pd.DataFrame(data=self.text_df.toarray(), columns=feats)

        self.X_train_df = pd.concat([self.imputed_df, TfIdf_df], axis=1)
        self.X_train_fea = list(self.X_train_df.columns.values)

    """
    Method for stratified sampling so that test containes 20% and train contains 80% of data
    from each class.
    """

    def statify_splits(self, num_splits=5, test_size=0.2):
        self.num_splits = num_splits
        self.test_size = test_size
        sss = StratifiedShuffleSplit(n_splits=self.num_splits, test_size=self.test_size, random_state=0)
        self.X_train_np, self.y_np = self.X_train_df.values, self.y.values
        self.splits = sss.split(self.X_train_np, self.y_np)
