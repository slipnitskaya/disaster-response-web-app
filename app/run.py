import json
import argparse

import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request

from plotly.graph_objs import Histogram

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from model.train_classifier import load_data, tokenize, MultiOutputClassifier
from model.train_classifier import RatioUpperExtractor, RatioNounExtractor, CountVerbExtractor

from typing import Optional, Tuple

app = Flask(__name__)


def parse_arguments() -> Tuple[str, str, Optional[str]]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Disaster Response / Web app')
    parser.add_argument('--path-to-database', type=str, default='data/disaster_responses.db')
    parser.add_argument('--path-to-model', type=str, default='../model/classifier.pkl')
    parser.add_argument('--table-name', type=str, required=False)
    args = parser.parse_args()

    return args.path_to_database, args.path_to_model, args.table_name


path_to_database, path_to_model, table_name = parse_arguments()

# load data
X, y, class_names = load_data(path_to_database)
df = pd.concat([X, y], axis=1)

# load model
model = joblib.load(path_to_model)


@app.route('/')
@app.route('/index')
def index():
    """
    Display visuals and receive user input text for model.
    """
    # extract data
    messages_per_cat = y.sum(axis=0)
    cats_per_message = y.sum(axis=1)

    # create visuals
    graphs = [
        {
            'data': [
                Histogram(x=cats_per_message)
            ],
            'layout': {
                'title': 'Number of Categories per Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Categories"
                }
            },
        },
        {
            'data': [
                    Histogram(x=messages_per_cat)
            ],
            'layout': {
                'title': 'Number of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Messages"
                }
            }
        }
    ]

    # encode graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
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
    main()
