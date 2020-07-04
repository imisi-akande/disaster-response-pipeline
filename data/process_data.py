import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load and merge messages & categories datasets

    inputs:
    messages_filepath: string -> Path to the csv file containing disaster messages dataset.
    categories_filepath: string -> Path to the csv file containing disaster categories dataset.

    outputs:
    df: dataframe. Dataframe containing combined disaster messages and categories datasets.
    """
    # Load Messages Dataset
    messages = pd.read_csv(messages_filepath)

    # Load Categories Dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """Function to clean categories data

    Args:
    df: dataframe. Dataframe containing combined disaster messages and categories datasets

    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    # create a df by splitting indiviual category
    categories = df['categories'].str.split(';', expand = True)

    # select first row
    row = categories.iloc[0]

    # obtain list of categories from each row by selecting up to the second to the last character
    category_colnames = row.transform(lambda x: x[:-2]).tolist()

    # rename all columns with categories
    categories.columns = category_colnames

    # Convert category numbers
    for column in categories:
        # set every value to be the last character of a string
        categories[column] = categories[column].transform(lambda x: x[-1:])

        # convert column from string to integer
        categories[column] = categories[column].astype(np.int)

    # Drop the categories field from dataframe
    df.drop('categories', axis = 1, inplace = True)

    # Concatenate dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis = 1)

    # Drop duplicates
    df.drop_duplicates(inplace = True)

    # Remove rows with a value of 2 from dataframe(0's and 1's are only allowed)
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    """ Function to save into SQLite database.

    inputs:
    df: dataframe. Dataframe containing cleaned version of combined disaster 
    message and categories data.

    database_filename: string. Path to the SQLite output database.

    outputs:
    None
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    """
    Main Data Processing function

    This function implements three major actions in creating the ETL pipeline:
        1) Data extraction from combining messages and categories files 
        2) Categories data cleaning and pre-processing
        3) Data loading to SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()