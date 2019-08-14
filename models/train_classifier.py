import sys
import pandas as pd
import numpy as np
import os
import pickle
import nltk
import datetime
import re
import sklearn
import helper_class
import pathlib
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from helper_class import Noun_POSCount,Verb_POSCount,Word_Count
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
sp = spacy.load('en_core_web_sm')
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    '''Method to load data from database
    
    Parameters:
    argument1 : Path of the database file
    
    Returns:
    a valid dataframe
    '''
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('disaster_messages',engine)
    print("Data loading complete...")
    df = df.fillna(0)
    X = df.message
    Y = df.iloc[:,4:]
    ##temp code
    #X = X.iloc[:10]
    #Y = Y.iloc[:10]
    return X,Y, Y.columns


def tokenize(text):
    '''Method to replace special characters, to lemmatize and stem words and return a single list with all the tokens in a corpus
    
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


def build_model():
    '''Method to create a new pipeline with collection of feature preprocessing steps and also a estimator
    
    Parameters:
    argument1 : no args

    Returns:
    a valid pipeline
   '''
    
    
    pipeline =  Pipeline([
    ("features", FeatureUnion([
        ("text", TfidfVectorizer(tokenizer=tokenize)),
        ("verb_count", Verb_POSCount()),
        ("noun_count", Noun_POSCount()),
        ("word_count",Word_Count())
        
    ])),
    ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''Method to evaluate the model and display f-scores for each category
    
    Parameters:
    argument1 : trained model
    argument2 : Data for evaluation
    argument3 : Labels for evaluation
    argument4 : List of categories

    Returns : None
    '''
    y_preds = model.predict(X_test)
    for pred, label, col in zip(y_preds.transpose(), Y_test.values.transpose(), Y_test.columns):
    #print(classification_report(label, pred))
        print(f"-------Report for {col}--------")
        print(classification_report(label, pred))
    return 


def save_model(model, model_filepath):
    '''Method to save the trained model to local filesystem
    
    Parameters:
    argument1 : trained model
    argument2 : full local path for the model to be saved

    Returns : None
    '''
    mpath = Path(model_filepath)
    parent = mpath.parent
    if (parent.exists()):
        print("path exists")
        print(f"Save the file {mpath.name} inside parent {mpath.parent}")
        pickle.dump(model, open(model_filepath, 'wb'))
    else:
        print("Path absent, creating it...")
        parent.mkdir(parents=True, exist_ok = True)
        print(f"Save the file {mpath.name} inside parent {mpath.parent}")
        pickle.dump(model, open(model_filepath, 'wb'))
    
    return 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(f"X shape is {X.shape}--Y shape is {Y.shape}")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Building model complete...go for fit...')
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Training complete...')
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()