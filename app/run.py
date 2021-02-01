import os
import sys
import json

import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request

from plotly.graph_objs import Bar, Histogram

from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath('..'))

from model.train_classifier import parse_arguments, load_data, tokenize
from model.train_classifier import RatioUpperExtractor, RatioNounExtractor, CountVerbExtractor


app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    """
    Display visuals and receive user input text for model.
    """
    # extract statistics
    messages_per_cat = y.sum(axis=0)
    cats_per_message = y.sum(axis=1)

    messages_per_cat = messages_per_cat.sort_values(ascending=False)
    categories, counts = zip(*messages_per_cat.to_dict().items())

    # create visuals
    graphs = [
        {
            'data': [
                Bar(x=categories, y=counts)
            ],
            'layout': {
                'title': 'Number of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Histogram(x=cats_per_message)
            ],
            'layout': {
                'title': 'Number of Categories per Message',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Number of Categories'
                }
            }
        }
    ]

    # encode graphs in JSON
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    Handle user query and display model results.
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(class_names, classification_labels))

    # render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    path_to_database, path_to_model = parse_arguments()

    # load data
    X, y, class_names = load_data(path_to_database)
    df = pd.concat([X, y], axis=1)

    # load model
    model = joblib.load(path_to_model)

    main()
