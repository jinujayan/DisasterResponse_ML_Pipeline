import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the messages and categories file into dataframe
    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        Two dataframes(messages,categories) -> Loaded data as Pandas DataFrames
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #Merge datasets on id field
    df = pd.merge(messages, categories, on='id')
    return messages,categories


def clean_data(messages_df,categories_df):
    """
    Merge and clean the data
    
    Arguments:
        messages_df,categories_df -> Loaded data frames
    Outputs:
        df -> Cleaned data frame
    """
    
    #Merge datasets on id field
    merged_df = pd.merge(messages_df, categories_df, on='id')
    categories_df = categories_df['categories'].str.split(';',expand=True)
    
    #Take a row of the dataframe and extract the column name only
    category_colnames = categories_df.iloc[0].apply(lambda x : x[:-2])
    categories_df.columns = category_colnames
   
    for column in categories_df:
        # set each value to be the last character of the string, this will give the numerical value at that cell
        categories_df[column] = categories_df[column].apply(lambda x : x[-1])
        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])
        ##Remove the column categories to enable add of newly transformed columns
    merged_df.drop(columns=['categories'], inplace= True)
    merged_df = pd.concat([merged_df,categories_df],axis=1)
    print(f"Number of duplicated rows are {merged_df.duplicated().sum()}")
    
    # drop duplicates
    merged_df = merged_df.drop_duplicates()
    print(f"Number of duplicated rows after cleaning are {merged_df.duplicated().sum()}")
    return merged_df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False)
    print("Data saved into disaster_messages table")
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        cleaned_df = clean_data(messages_df,categories_df)
        print(f"Cleaning complete...shape is {cleaned_df.shape}")
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(cleaned_df, database_filepath)
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