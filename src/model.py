import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_preprocessed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('/home/emirhan/anomaly-detection/data/X_train.csv')
    X_test = pd.read_csv('/home/emirhan/anomaly-detection/data/X_test.csv')
    y_train = pd.read_csv('/home/emirhan/anomaly-detection/data/y_train.csv')
    y_test = pd.read_csv('/home/emirhan/anomaly-detection/data/y_test.csv')
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest Classifier"""
    # Encode target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train['Attack_type'])
    y_test_encoded = le.transform(y_test['Attack_type'])
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train_encoded)
    
    # Predict and evaluate
    y_pred = rf_classifier.predict(X_test)
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test_encoded, y_pred, 
                                target_names=le.classes_))
    
    # Save the model
    joblib.dump(rf_classifier, '/home/emirhan/anomaly-detection/src/random_forest_model.joblib')
    joblib.dump(le, '/home/emirhan/anomaly-detection/src/label_encoder.joblib')
    
    return rf_classifier, le

def train_deep_learning_model(X_train, X_test, y_train, y_test):
    """Train Deep Learning Model (Neural Network)"""
    # Encode target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train['Attack_type'])
    y_test_encoded = le.transform(y_test['Attack_type'])
    
    # One-hot encode the target
    y_train_onehot = to_categorical(y_train_encoded)
    y_test_onehot = to_categorical(y_test_encoded)
    
    # Build Neural Network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(y_train_onehot.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train_onehot, 
        validation_split=0.2, 
        epochs=50, 
        batch_size=32
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Deep Learning Model Test Accuracy: {test_accuracy}")
    
    # Save the model
    model.save('/home/emirhan/anomaly-detection/src/deep_learning_model.h5')
    
    return model

def main():
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train Random Forest Model
    rf_model, label_encoder = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Train Deep Learning Model
    dl_model = train_deep_learning_model(X_train, X_test, y_train, y_test)
    
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    main()
