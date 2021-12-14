# Importing Main Libraries
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import wordnet
import warnings
import numpy as np
import re
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

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


class DataPreprocessing(object):
    """
    Loading Data into Pandas DataFrame.
    Removing features of the data which are less than 5% available or which are more than 95% missing.
    Removeing Noisy cols i.e. id, member_id, url. These features are random, unique and unimportant.
    target = "loan_status"
    Cleaning columns int_rate and revol_util. E.g. Converting 10.25% string to float 10.25
    """

    def __init__(self, datafile=None, threshold_percent=0.05, noisy_cols=None, target_col=None):
        super(DataPreprocessing, self).__init__()
        self.datafile = datafile
        self.data = pd.read_csv(self.datafile)
        self.threshold_percent = threshold_percent
        self.length = len(self.data)
        self.noisy_cols = noisy_cols
        self.target_col = target_col

        print("Length of dataset = " + str(self.length))
        threshold = self.threshold_percent * self.length
        self.y = self.data[self.target_col]
        self.data = self.data.drop([self.target_col], axis=1)
        self.data = self.data.drop(self.noisy_cols, axis=1)

        self.removingColsWithNans(threshold)

        print("\nFrequency of Loan Status " + str(dict(Counter(self.y))) + "\n")
        print("\nDataset Shape : " + str(self.data.shape))
        self.data['int_rate'] = self.data['int_rate'].str.replace('%', '').astype(float)
        self.data['revol_util'] = self.data['revol_util'].str.replace('%', '').astype(float)

    """
    Removing features of the data which are less than 5% available or which are more than 95% missing.  
    Large number Missing features are similar to find their values by their classfication or regression from
    other features.
    """

    def removingColsWithNans(self, threshold):
        print(
            "\nFeatures contains less than " + str(self.threshold_percent * 100) + "% of whole data has been Removed:")
        removedcols, retaincols, cols = [], [], list(self.data.columns.values)

        for c in cols:
            if self.length - self.data[c].isnull().sum() > threshold:
                retaincols.append(c)
            else:
                removedcols.append(c)

        print("\nNumber of Removed Columns " + str(len(removedcols)))
        print(removedcols)
        print("\nNumber of Retained Columns " + str(len(retaincols)))
        print(retaincols)

        self.data = self.data[retaincols]

    """
    Expanding the date column into 2 numeric features columns.
    E.g Jan-31 expanded to 2 numeric columns i.e 1 (month) and 31 (day)
    E.g Jan-89 expanded to 2 numeric columns i.e 89 (year) and 1 (month)
    """

    def expandingCol(self, colname):
        month = colname + '_Month'
        day_or_year = colname + '_Year' if colname == 'earliest_cr_line' else colname + '_Day'
        cols = {0: month, 1: day_or_year}
        self.data = self.data.join(self.data[colname].str.split('-', 1, expand=True).rename(columns=cols))
        self.data = self.data.drop([colname], axis=1)
        self.data[day_or_year] = self.data[day_or_year].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
        self.data[month] = self.data[month].apply(lambda x: int(month_d[str(x).lower()]) if pd.notnull(x) else np.nan)

    """
    Creating the dataFrame to describe about Feature, Number_Missing_Values and DataType
    """

    def dataDescribe(self):
        description_df = pd.DataFrame(columns=['Feature', 'Number_Missing_Values', 'DataType'])
        description_df['Feature'] = list(self.data.columns.values)
        description_df['Number_Missing_Values'] = self.data.isnull().sum().values
        description_df['DataType'] = self.data.dtypes.values
        print("\nDataset Description: ")
        print(description_df)

    """
    Separating into  2 dataFrames i.e numeric type and text type
    """

    def separating_Numeric_and_CategoryTextFeatures(self):
        self.numeric_df = self.data.select_dtypes(include=[np.number])
        self.cat_df = self.data.select_dtypes(include=[object])
        print("\nNumeric and Category Data Shapes: ")
        print(self.numeric_df.shape)
        print(self.cat_df.shape)

    """
    Method to get Part of Speech Tag for particular word
    """

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    """
    Method for Lemmatization of Text using Part of Speech Tag for particular word
    """

    def lemmatize_text(self, text):
        return [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]

    """
    Method for Porter Stemmatization of Text
    """

    def stemmatize_text(self, text):
        return [PorterStemmer().stem(item) for item in nltk.word_tokenize(text)]

    """
    Method for cleaning of Text. It include removing invalid numerical numbers in text, punctatations and invalid
    english Vocab Words. It will be explained separately on their separate methods
    """

    def cleaning_text(self):
        self.cat_df['desc'] = self.cat_df['desc'].apply(lambda x: re.sub(r'\d+', '', str(x)))
        self.cat_df['desc'] = self.cat_df['desc'].replace('br', '', regex=True)
        self.cat_df['desc'] = self.cat_df['desc'].apply(
            lambda s: s.translate(str.maketrans('', '', string.punctuation)))
        self.cat_df['desc'] = self.cat_df['desc'].apply(lambda s: self.cleanstring_Removing_br(s))
        self.cat_df['desc'] = self.cat_df['desc'].apply(lambda s: self.removeInvalidWords(s))

    """
    Method for cleaning of Text i.e. removing invalid english Vocab Words. Method Checks whether word exist
    in Corpus set of words that makes sense. E.g. 'abdidebr' is not an english word so it is discarded.
    This process highly time expensive.
    """

    def removeInvalidWords(self, sent):
        words = set(nltk.corpus.words.words())
        return ' '.join(w for w in nltk.wordpunct_tokenize(sent) if w.lower() in words or not w.isalpha())

    """
    Method for cleaning Text. Text containes number of words that contains 'br' keyword in the end or
    'br' twice in the beginning. This may occur due to incorrect text parsing. This module removes 
    these kind of invalid words. E.g. 'abroadbr', 'brbramong'
    """

    def cleanstring_Removing_br(self, s):
        clean_s = [wrd[:-2] if wrd[-2:] == 'br' or wrd[:4] == 'brbr' else wrd for wrd in s.split(' ')]
        return ' '.join(clean_s)

    """
    Method for convering Text to vectorizer. TfIdf is used and Other vectorizer methods can also be used.
    In this Method, Lemmatization, removing stop words and top 2000 features have beed considered 
    Before TF_Idf, around 4500  unique features were present in text. But top 2000 features have beed 
    considered to reduce dimensionality.
    """

    def TfIdfVectorizer_TextTransformation(self):
        self.cleaning_text()
        fn = getattr(self, 'lemmatize_text')
        print("\nTextual Dataset shape after TfIdfVectorizer:")
        vectorizer = TfidfVectorizer(tokenizer=fn, max_df=0.95,
                                     min_df=3, stop_words='english', lowercase=True, max_features=2000)
        self.X_cat = vectorizer.fit_transform(self.cat_df['desc'])
        print(self.X_cat.shape)
