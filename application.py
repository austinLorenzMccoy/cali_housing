from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
application = Flask(__name__)
app = application

# Function to load the saved model
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to load the saved scaler
def load_scaler():
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Load the model and scaler at the start
model = load_model()
scaler = load_scaler()

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from the request
        input_features = [
            data['longitude'],
            data['latitude'],
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income']
        ]

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale the input features
        input_scaled = scaler.transform(input_array)
        
        # Make the prediction
        prediction = model.predict(input_scaled)

        # Return the prediction as JSON
        return jsonify({'price': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
