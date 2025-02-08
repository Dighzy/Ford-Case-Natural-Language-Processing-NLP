import re
import json
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
custom_model = load_model('./models/best_model.h5')

# Load car models list
with open("./models/models_params.json", "r") as file:
    car_models = json.load(file)['models']

# Load trained scaler
with open("./models/scaler_params.json", "r") as file:
    scaler_params = json.load(file)
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_params["mean"])
scaler.scale_ = np.array(scaler_params["scale"])

# Load pre-trained TF-IDF and LDA models
tfidf_vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
lda_model = joblib.load("./models/lda_model.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text).lower().strip()
    return ' '.join(text.split())

# Function to generate BERT embeddings
def get_model_embedding(text):
    if not text:
        return -0.5  # Placeholder for missing embedding

    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy() 

# Function to extract additional features
def extract_additional_features(text, model):
    # Preprocess text
    processed_text = preprocess_text(text)
    word_count = len(processed_text.split())
    char_count = len(processed_text)
    sentiment = TextBlob(processed_text).sentiment.polarity

    # Generate TF-IDF vector and topic features
    tfidf_vector = tfidf_vectorizer.transform([processed_text])
    topic_vector = lda_model.transform(tfidf_vector).flatten()

    # Assign topic values to separate variables
    topic_1, topic_2 = topic_vector[0], topic_vector[1]

    # Generate model embedding (scalar for simplicity)
    model_embedding = get_model_embedding(model)
    model_embedding= np.array(model_embedding.tolist())

    # Combine features into a structured dictionary
    features = {
        "model_embedding": model_embedding,
        "word_count": word_count,
        "char_count": char_count,
        "sentiment": sentiment,
        "topic_1": topic_1,
        "topic_2": topic_2
    }

    # Convert features to DataFrame for consistency
    feature_df = pd.DataFrame([features])
    
    # Normalize the features
    scaled_features = scaler.transform(feature_df)

    return scaled_features

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    text = input_data.get('summary')
    model_name = input_data.get('model')

    if not text or not model_name:
        return jsonify({"error": "Both 'summary' and 'model' must be provided."}), 400
    
    #print('here1', text)

    # Generate features
    embedding = get_model_embedding(text)
    embedding= np.array(embedding.tolist())
    print('here', embedding)
    #embedding = np.expand_dims(embedding, axis=0)  # Ensure shape (1, 768)

    additional_features = extract_additional_features(text, model_name)  # Already (1, 6)

    # Make predictions
    prediction = custom_model.predict({
        "embedding_input": embedding,
        "additional_input": additional_features
    })

    # Format response
    categories = ["Engine and Performance", "Electrical System", "Other Problems", "Structure and Control", "Safety and Brakes"]
    predicted_categories = [categories[i] for i, prob in enumerate(prediction[0]) if prob > 0.5]

    response = {
        "summary": text,
        "model": model_name,
        "predicted_categories": predicted_categories,
        "probabilities": prediction[0].tolist()
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
