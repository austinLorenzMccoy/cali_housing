import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Function to load the dataset
def load_data():
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y = pd.Series(california_housing.target, name='MedHouseVal')
    return X, y

# Preprocess the data (splitting and scaling)
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Save the trained model using pickle
def save_model(model):
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved to model.pkl")

# Save the scaler using pickle
def save_scaler(scaler):
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Scaler saved to scaler.pkl")

# Load, preprocess, train, and save model/scaler
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train the model
    model = train_model(X_train, y_train)

    # Save the model and scaler to disk
    save_model(model)
    save_scaler(scaler)
