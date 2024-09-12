from flask import Flask, render_template, request, jsonify
from model import load_data, preprocess_data, train_model, predict_price

application = Flask(__name__)
app = application

# Load, preprocess the data, and train the model
X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
model = train_model(X_train, y_train)

# Route for the homepage
@app.route('/')

def index():
    return render_template('index.html')  # Load the homepage

if __name__ == '__main__':
    app.run(debug=True)  # Run the app
