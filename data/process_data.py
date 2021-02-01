import os
import argparse

import pandas as pd

from typing import Tuple
from typing import Optional

from sqlalchemy import create_engine


PATH_TO_MESSAGES: Optional[str] = None
PATH_TO_CATEGORIES: Optional[str] = None
PATH_TO_DATABASE: Optional[str] = None


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load the messages and categories datasets and merge them.

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how='inner', on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data.
    """

    # split the categories column into separate columns
    categories = df.categories.str.split(pat=';', expand=True)

    # rename columns
    category_colnames = categories.iloc[0, :].apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # select the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # convert category values to binary
    categories = (categories > 0).astype(int)

    # concatenate the original dataframe with the new categories
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1, join='inner')

    # drops duplicates
    df = df.drop_duplicates()

    return df


def save_data(df: pd.DataFrame, path_to_database: str = 'disaster_responses.db', table_name: Optional[str] = None):
    """
    Save the cleaned data into a SQLite database.
    """
    if table_name is None:
        table_name, _ = os.path.splitext(os.path.basename(path_to_database))

    engine = create_engine(f'sqlite:///{path_to_database}')
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def parse_arguments() -> Tuple[str, str, str]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Disaster Response / Data Processing')
    parser.add_argument('-m', '--path-to-messages', type=str)
    parser.add_argument('-c', '--path-to-categories', type=str)
    parser.add_argument('-d', '--path-to-database', type=str)
    args = parser.parse_args()

    return args.path_to_messages, args.path_to_categories, args.path_to_database


def main():
    global PATH_TO_MESSAGES, PATH_TO_CATEGORIES, PATH_TO_DATABASE

    PATH_TO_MESSAGES, PATH_TO_CATEGORIES, PATH_TO_DATABASE = parse_arguments()

    print(f'Loading data...\n\tMessages: {PATH_TO_MESSAGES}\n\tCategories: {PATH_TO_CATEGORIES}')
    df = load_data(PATH_TO_MESSAGES, PATH_TO_CATEGORIES)

    print('Cleaning data...')
    df = clean_data(df)

    print(f'Saving data...\n\tDatabase: {PATH_TO_DATABASE}')
    save_data(df, PATH_TO_DATABASE)

    print('Cleaned data saved to database.')


if __name__ == '__main__':
    main()
