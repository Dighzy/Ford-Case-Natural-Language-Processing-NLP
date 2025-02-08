
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from process_data import preprocess_text, get_model_embedding, get_sentiment_and_count, get_topics, get_processed_features

def get_model(embeddings, additional_features):
    
    # The model uses two input branches: embeddings from complaint texts and structured features. 
    # Gaussian noise and dropout improve generalization, while dense layers with batch normalization stabilize learning.
    # The dual-input approach enhances predictions by leveraging both semantic and contextual information. 
    # Optimizations like learning rate scheduling, early stopping, and model checkpointing prevent overfitting.


    # Define inputs
    embedding_input = Input(shape=(embeddings.shape[1],), name="embedding_input")
    additional_input = Input(shape=(additional_features.shape[1],), name="additional_input")

    # Add Gaussian Noise to the embedding input
    x1 = GaussianNoise(0.01)(embedding_input)  # Adding noise

    # Embedding branch with batch normalization and L2 regularization
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    # Additional features branch with batch normalization and L2 regularization
    x2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(additional_input)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)

    # Combine both branches
    combined = Concatenate()([x1, x2])
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)

    # Output layer for multi-label classification
    output = Dense(5, activation='sigmoid')(combined)

    # Define the model
    model = Model(inputs=[embedding_input, additional_input], outputs=output)

    optimizer = Adam(learning_rate=0.00015)

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall','Precision'])

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

    # Defining the model

    model = get_model(embeddings, additional_features)

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the path to save the best model
    best_model_path = "models/best_model_teste.h5"

    # Add the ModelCheckpoint callback to save the best model
    model_checkpoint = ModelCheckpoint(
        filepath=best_model_path,  # Path to save the model
        monitor='val_loss',        # Monitor the validation loss
        save_best_only=True,       # Save only when the validation loss improves
        mode='min',                # Save when the monitored metric is minimized
        verbose=1                  # Print message when saving the model
    )


    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',  # Métrica a ser monitorada
        factor=0.25,          # Reduz o learning rate pela metade
        patience=3,          # Reduz o LR após 3 épocas sem melhoria
        min_lr=1e-6          # Limite mínimo para o learning rate
    )
    # Train the model
    history = model.fit(
        {"embedding_input": X_emb_train, "additional_input": X_add_train},
        y_train,
        validation_data=({"embedding_input": X_emb_test, "additional_input": X_add_test}, y_test),
        epochs=200,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )


