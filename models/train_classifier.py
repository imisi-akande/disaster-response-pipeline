import os
import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, make_scorer)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Function to load data

    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> target DataFrame
        category_names -> List of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace(".db","") + "_table"
    print(table_name, 'this is')
    df = pd.read_sql_table(table_name, engine)

    # Drop child_alone field because its all occupied by zeros only
    df = df.drop(['child_alone'],axis=1)

    # Replace 2 with 1 to consider it a valid response(binary).
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text, url_place_holder_string="urlplaceholder"):
    """
    function to tokenize and normalize text data

    Arguments:
        text -> messages to be tokenized and normalized
    Output:
        normalized -> List of tokens extracted and normalized from the messages
    """

    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    #Lemmatizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    normalized = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return normalized

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Function for building pipeline

    The Scikit ML Pipeline processes text messages and apply a classifier.
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate model

    Arguments:
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories

    Output
        The method prints out the precision, recall and f1-score
    """
    Y_pred = model.predict(X_test)

    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('The average overall accuracy {0:.2f}% \n'.format(overall_accuracy * 100))

    # Print the overall classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)

    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))



def save_model(pipeline, pickle_filepath):
    """
    Function to save the model to disk

    Arguments:
        model: The model to be saved
        model_filepath: The filepath where the model is to be saved
    OUTPUT
        This method will save the model as a pickled file.
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))


def main():
    """
    Train Classifier Main function

    This function implements four major actions in the Machine Learning Pipeline:
        1) Extract data from SQLite DB
        2) Train ML model on the training dataset
        3) Evaluate the model performance on test dataset
        4) Save trained model as a pickled file

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

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
