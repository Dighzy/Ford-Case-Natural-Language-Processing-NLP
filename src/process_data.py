import pandas as pd
import numpy as np
import tensorflow as tf
import re
import nltk
import torch
import joblib
import json

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from textblob import TextBlob

# Baixar recursos necessários
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing function

def preprocess_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    
    text = text.lower()
    
    # Tokenization: Split the text into individual words
    words = word_tokenize(text)
    
    # Remove stopwords (common words like 'the', 'and', etc.)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization: Convert words to their base or root form (e.g., 'running' to 'run')
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def get_model_embedding(text):
    #In these case i decided to use bert because its a more robust model that generates contextual embeddings.
    #This is important for analyzing complaints, as it helps capture the full meaning of the text and identify patterns or issues more accurately
    #I chose BERT over other models because its lighter while still providing strong contextual embeddings without needing a complete transformer

     # Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    """Converts text into a BERT embedding"""
    if pd.isna(text):
        return torch.zeros(512)  
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_topics(df):
    
    # Generatinhg additional features using a topic modeling 
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_matrix = tfidf.fit_transform(df["processed_summary"])

    #Model identifies two latent topics in the dataset.
    topic_model = joblib.load("models/lda_model.pkl")
    topic_features = topic_model.fit_transform(tfidf_matrix)


    topic_df = pd.DataFrame(topic_features, columns=[f"topic_{i+1}" for i in range(topic_features.shape[1])])
    df = pd.concat([df, topic_df], axis=1)

    return df


def get_sentiment_and_count(df):
    
    # Calculating the word count and character count for the processed summary
    df["word_count"] = df["processed_summary"].apply(lambda x: len(x.split()))  # Count the number of words
    df["char_count"] = df["processed_summary"].apply(len)  # Count the number of characters

    # Calculating the sentiment polarity for each processed summary
    df['sentiment'] = df['processed_summary'].apply(lambda x: TextBlob(x).sentiment.polarity)
     # 1.0 → Positive, 0.0 → Neutral, -1.0 → Negative

    return df

def get_processed_features(df):
    
    # Load trained scaler
    with open("models/params/scaler_params.json", "r") as file:
        scaler_params = json.load(file)

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])

    # Convert embeddings and category binary column to NumPy arrays
    embeddings = np.array(df["summary_embedding"].tolist())
    df['model_embedding'] = np.array(df["model_embedding"].tolist())

    # Normalize additional features
    additional_features = df[['model_embedding','word_count', 'char_count', 'sentiment', 'topic_1', 'topic_2']]
    additional_features = scaler.transform(additional_features)

    return embeddings, additional_features


if __name__ == "__main__":
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv('data/raw/full_data_2021_FORD.csv')
    df_final = df.head(5).copy()

    # Getting model types
    models = df_final['Model'].unique().tolist()
    model_data = {
        "models":models
    }
    with open('models/params/models_params.json', 'w') as f:
        json.dump(model_data, f)
    
    # Text preprocessing function
    df_final['processed_summary'] = df_final['summary'].apply(preprocess_text)

    # Get embeddings
    df_final['summary_embedding'] = df_final['summary'].apply(get_model_embedding)
    df_final['model_embedding'] = df_final['Model'].apply(get_model_embedding)

    # Get sentiment and topics
    df_final = get_sentiment_and_count(df_final)
    df_final = get_topics(df_final)

    # Processing and normalazing the embedding and additional_features
    embeddings, additional_features = get_processed_features(df_final)

    # Verify the shapes of the data
    print("embeddings shape:", embeddings.shape) 
    print("additional_features shape:", additional_features.shape) 

    print("embeddings:", embeddings)
    print("additional_features:", additional_features)
