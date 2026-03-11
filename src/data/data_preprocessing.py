import numpy as np
import pandas as pd
import os
import sys
import re
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    """Download an NLTK resource only if it is not already installed."""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, quiet=True)

def preprocess_dataframe(df, col='text'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.    
    """


    # Intialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    def  preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    # Apply preprocessing to the specified column
    df[col] = df[col].apply(preprocess_text)

    # Remove small sentences (less than 3 words)
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)
    # Drop rows with NaN values
    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df
def main():


    try:

        # Fetch the data from data/raw
        ensure_nltk_resource('corpora/wordnet', 'wordnet')
        ensure_nltk_resource('corpora/stopwords', 'stopwords')

        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info("Data fetched successfully")

        #Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')


        # store the data inside data/processed

        data_path = os.path.join("./data", "interim")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logging.info("Data stored successfully")    
    except Exception as e:
        logging.error("Error in data preprocessing: %s", str(e))
        raise e     


if __name__ == '__main__':
    main()
