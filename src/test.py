
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model

from process_data import preprocess_text, get_model_embedding, get_sentiment_and_count, get_topics, get_processed_features

if __name__ == "__main__":

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    df = pd.read_csv('data/raw/full_data_2021_FORD.csv')
    df_final = df.head(5).copy()

    df_final['processed_summary'] = df_final['summary'].apply(preprocess_text)
    df_final['summary_embedding'] = df_final['summary'].apply(get_model_embedding)
    df_final['model_embedding'] = df_final['Model'].apply(get_model_embedding)


    df_final = get_sentiment_and_count(df_final)
    df_final = get_topics(df_final)

    embeddings, additional_features = get_processed_features(df_final)

    custom_model = load_model('models/best_model.h5')

    # Make predictions
    prediction = custom_model.predict({
        "embedding_input": embeddings,
        "additional_input": additional_features
    })

    threshold = 0.5
    binary_predictions = (prediction >= threshold).astype(int)

    categories = ["Engine and Performance", "Electrical System", "Other Problems", "Structure and Control", "Safety and Brakes"]
    formatted_predictions = []

    for row in binary_predictions:
        predicted_categories = [categories[i] for i, c in enumerate(row) if c == 1]
        formatted_predictions.append(predicted_categories)

    df_final['binary_predictions'] = [list(row) for row in binary_predictions]
    df_final['predictions'] = formatted_predictions
    df_final = df_final[['Model','summary','binary_predictions','predictions']]

    print(df_final.head(5))

