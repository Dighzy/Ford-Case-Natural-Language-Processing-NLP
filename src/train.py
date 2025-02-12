
import pandas as pd
import tensorflow as tf
import numpy as np
import ast
import json
import joblib
import random 

import keras_tuner as kt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

def get_model_tuner(X_emb_train, X_add_train):
    # The model uses two input branches: embeddings from complaint texts and structured features. 
    # Gaussian noise and dropout improve generalization, while dense layers with batch normalization stabilize learning. 
    # The dual-input approach enhances predictions by leveraging both semantic and contextual information. 
    # Optimizations like learning rate scheduling, early stopping, and model checkpointing prevent overfitting.
    # Additionally, Bayesian optimization was used to select the best hyperparameters, ensuring an efficient and effective search for optimal model configurations.


    # Define the model-building function for the tuner
    def build_model(hp):
        # Define inputs
        embedding_input = Input(shape=(X_emb_train.shape[1],), name="embedding_input")
        additional_input = Input(shape=(X_add_train.shape[1],), name="additional_input")

        # Add Gaussian noise as a regularization technique
        x1 = GaussianNoise(hp.Float('noise', 0.005, 0.02))(embedding_input)

        # Dense layer for embeddings (input from X_emb_train)
        x1 = Dense(hp.Int('units_x1', 64, 256, step=32), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x1', [0.001, 0.01, 0.1])))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(hp.Float('dropout_x1', 0.2, 0.5))(x1)

        # Dense layer for additional features (input from X_add_train)
        x2 = Dense(hp.Int('units_x2', 32, 128, step=16), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x2', [0.001, 0.01, 0.1])))(additional_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(hp.Float('dropout_x2', 0.2, 0.5))(x2)

        # Concatenate the two blocks of layers
        combined = Concatenate()([x1, x2])
        combined = Dense(hp.Int('units_combined', 16, 64, step=16), activation='relu', kernel_regularizer=l2(hp.Choice('l2_combined', [0.001, 0.01, 0.1])))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(hp.Float('dropout_combined', 0.2, 0.5))(combined)

        # Output layer for multi-label classification (sigmoid activation for each label)
        output = Dense(5, activation='sigmoid')(combined)

        # Create the model
        model = Model(inputs=[embedding_input, additional_input], outputs=output)

        # Compile the model with Adam optimizer and binary cross-entropy loss for multi-label classification
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

        return model

    # Set up the tuner for Bayesian optimization
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',  # Optimize based on validation loss
        max_trials=20,  # Number of trials (different configurations)
        executions_per_trial=1,
        directory="models/tuner_results", 
        project_name="bayesian_opt_nlp"  
    )

    return tuner

def get_topic_features(df):
    # Generatinhg additional features using a topic modeling 
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["processed_summary"])

    #Model identifies two latent topics in the dataset.
    topic_model = LatentDirichletAllocation(n_components=2, random_state=42)
    topic_features = topic_model.fit_transform(tfidf_matrix)

    topic_df = pd.DataFrame(topic_features, columns=[f"topic_{i+1}" for i in range(topic_features.shape[1])])
    df_final = pd.concat([df, topic_df], axis=1)

    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump(topic_model, "models/lda_model.pkl")

    return df_final

def  get_processed_features(df):
        # Converting columns 
    df["category_binary"] = df["category_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["model_embedding"] = df["model_embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["summary_embedding"] = df["summary_embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Convert embeddings and category binary column to NumPy arrays
    X_embeddings = np.array(df["summary_embedding"].tolist())
    y_category_binary = np.array(df["category_binary"].tolist())  # Ensure it's a 2D array
    df['model_embedding'] = np.array(df["model_embedding"].tolist())

    # Normalize additional features
    additional_features = df[['model_embedding','word_count', 'char_count', 'sentiment', 'topic_1', 'topic_2']]
    scaler = StandardScaler()
    X_additional_features = scaler.fit_transform(additional_features)

    print("Scaler mean:", scaler.mean_)
    print("Scaler scale:", scaler.scale_)

    # Define mean and scale values
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
}
    # Save to JSON file
    with open('models/params/scaler_params.json', 'w') as f:
        json.dump(scaler_data, f)

    return X_embeddings, X_additional_features, y_category_binary


def set_test_df(X_emb_test, X_add_test, y_test):
    # Convert arrays to DataFrames
    df_emb_test = pd.DataFrame(X_emb_test)
    df_add_test = pd.DataFrame(X_add_test)

    # Convert multi-label `y_test` into lists (each row is a list of labels)
    y_test_lists = [list(row) for row in y_test]

    # Combine features and labels into a single DataFrame
    df_test_final = pd.DataFrame({
        'embeddings': df_emb_test.values.tolist(),  # Convert rows of feature data to  list
        'additional_features': df_add_test.values.tolist(),  # Convert additional features to list
        'category_binary': y_test_lists  # Store labels as lists
    })

    # Save to CSV
    df_test_final.to_csv("data/processed/test_data.csv", index=False)

if __name__ == "__main__":

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(4)

    # Loading the data

    df_final = pd.read_csv('data/processed/df_final_model.csv')

    df_final  = get_topic_features(df_final)

    X_embeddings, X_additional_features, y_category_binary = get_processed_features(df_final)


     # Split data into train and test sets
    X_emb_train, X_emb_val, X_add_train, X_add_val, y_train, y_val = train_test_split(
        X_embeddings, X_additional_features, y_category_binary, test_size=0.2, random_state=42
    )

    # Take the first 10 rows from the validation set for the test set
    X_emb_test = X_emb_val[:10]
    X_add_test = X_add_val[:10]
    y_test = y_val[:10]

    # Remove those 10 rows from the validation set
    X_emb_val = X_emb_val[10:]
    X_add_val = X_add_val[10:]
    y_val = y_val[10:]

    # Verify the shapes of the data
    print("X_emb_train shape:", X_emb_train.shape)  # Should match input shape for embedding input
    print("X_add_train shape:", X_add_train.shape)  # Should match input shape for additional input
    print("y_train shape:", y_train.shape)       # Should be (num_samples, 5) if 5 labels are expected

    set_test_df(X_emb_test, X_add_test, y_test)
    
    # Defining the model

    tuner = get_model_tuner(X_emb_train, X_add_train)

        # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    best_model_path = "models/best_model.h5"
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, min_lr=1e-6)

    # Start searching for the best hyperparameters
    tuner.search(
        {"embedding_input": X_emb_train, "additional_input": X_add_train},
        y_train,
        validation_data=({"embedding_input": X_emb_val, "additional_input": X_add_val}, y_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )

    # Get the best hyperparameters from the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters found: {best_hps.values}")

    # Train the final model using the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        {"embedding_input": X_emb_train, "additional_input": X_add_train},
        y_train,
        validation_data=({"embedding_input": X_emb_val, "additional_input": X_add_val}, y_val),
        epochs=200,  
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )


