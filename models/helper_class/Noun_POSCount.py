import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download(['stopwords'])
import en_core_web_sm
sp = spacy.load('en_core_web_sm')


class Noun_POSCount(BaseEstimator, TransformerMixin):
    def vnoun_count(self, text):
        counter = 0
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            spc_sentence = sp(' '.join(self.tokenize(sentence)))
            counter += [word.pos_ for word in spc_sentence].count('NOUN')
                    
        return counter

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.vnoun_count)
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(pd.DataFrame(count))
        return scaled_df
    
    def tokenize(self,text):
        '''Method to replace special characters, to lemmatize and stem words and return a single list with all the tokens in a      corpus
    
    Parameters:
    argument1 (pandas series): The column data as a series

    Returns:
    a valid list
    '''
        words = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        words = word_tokenize (words)
        eng_stop_words = stopwords.words("english")
        words = [word for word in words if word not in eng_stop_words]
        lemmetized = [WordNetLemmatizer().lemmatize(word) for word in words]
        words = [PorterStemmer().stem(word) for word in lemmetized]
        return words