import os
import re
import pickle
import logging
import argparse

import nltk

import numpy as np
import pandas as pd

import sklearn.pipeline as skpipe
import sklearn.decomposition as skdec
import sklearn.preprocessing as skprep
import sklearn.model_selection as skms
import sklearn.feature_extraction.text as skfet

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine

from typing import List, Tuple, Optional

nltk.download(['stopwords', 'wordnet', 'averaged_perceptron_tagger'])

STOP_WORDS = [w for w in stopwords.words('english')]

RANDOM_SEED = 42


def parse_arguments() -> Tuple[str, str]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Disaster Response / Message classification')
    parser.add_argument('-d', '--path-to-database', type=str)
    parser.add_argument('-M', '--path-to-model', type=str)
    args = parser.parse_args()

    return args.path_to_database, args.path_to_model


def load_data(path_to_database: str, table_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Load the data from the SQLite database and split it into feature and target variables.
    """
    if table_name is None:
        table_name, _ = os.path.splitext(os.path.basename(path_to_database))

    engine = create_engine(f'sqlite:///{path_to_database}')
    df = pd.read_sql_table(table_name, con=engine)

    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)

    class_names = y.columns.tolist()

    return X, y, class_names


def tokenize(text):
    """
    Clean, normalize, lemmatize, and tokenize the text.
    """
    # normalize the text
    url_expression = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_expression, '', text)

    # tokenize the text
    tokens = word_tokenize(text)

    # remove stop words and punctuations
    tokens = [token for token in tokens if token not in STOP_WORDS]

    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token]

    return tokens


class RatioUpperExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the ratio of uppercase words.
    """
    @staticmethod
    def get_upper_ratio(text):
        words_total = [word.strip(' ') for word in text.split(' ')]

        if not words_total:
            return 0.0

        words_upper = [word for word in words_total if word == word.upper() and len(word) > 1]
        return 1.0 * len(words_upper) / len(words_total)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(list(map(self.get_upper_ratio, X))).reshape(-1, 1)


class CountVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the number of verbs.
    """
    @staticmethod
    def count_verbs(text):
        tokens = tokenize(text)

        if not tokens:
            return 0.0

        return len([word for word, tag in pos_tag(tokens) if tag.startswith('VB')])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(list(map(self.count_verbs, X))).reshape(-1, 1)


class RatioNounExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the ratio of nouns.
    """
    @staticmethod
    def get_noun_ratio(text):
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        pos_tags = pos_tag(tokens)
        nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
        return len(nouns) / len(pos_tags)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(list(map(self.get_noun_ratio, X))).reshape(-1, 1)


def build_model():
    """
    Build an NLP pipeline for multi-label text classification.
    """
    # text processing and model pipeline
    pipeline = skpipe.Pipeline([
        ('nlp', skpipe.FeatureUnion([
            ('tfif', skpipe.Pipeline([('feat', skfet.TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize)),
                                      ('lsa', skdec.TruncatedSVD(n_components=200, algorithm='arpack'))])),
            ('uppr', RatioUpperExtractor()),
            ('verb', CountVerbExtractor()),
            ('noun', RatioNounExtractor())
        ])),
        ('norm', skprep.StandardScaler()),
        ('clf', MLPClassifier(
            activation='logistic', learning_rate='adaptive', early_stopping=True, random_state=RANDOM_SEED, verbose=1
        ))
    ])

    # define grid search parameters
    params = {
        'clf__learning_rate_init': [5e-3, 7.5e-3, 1e-2],
        'clf__hidden_layer_sizes': [(100), (200,), (300,)]
    }
    # instantiate GridSearchCV object
    cv = skms.GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    return cv


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names: List) -> None:
    """
    Display performance metrics for the trained model.
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))


def save_model(model, path_to_model: str) -> None:
    """
    Export the trained model as a pickle file.
    """
    with open(path_to_model, 'wb') as file:
        pickle.dump(model, file)


def main():
    path_to_database, path_to_model = parse_arguments()

    print(f'Loading data...\n\tDatabase: {path_to_database}')
    X, y, class_names = load_data(path_to_database)
    X, y = X.values, y.values

    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    print('Building model...')
    model = build_model()

    print('Training model...')

    try:
        model.fit(X_train, y_train)
    except Exception as ex:
        logging.exception(ex)
        print('Grid search failed.')

    print('Evaluating model...')
    evaluate_model(model.best_estimator_, X_test, y_test, class_names)

    print('Saving model...\n\tModel: {}'.format(path_to_model))
    save_model(model.best_estimator_, path_to_model)

    print('Trained model saved.')


if __name__ == '__main__':
    main()
