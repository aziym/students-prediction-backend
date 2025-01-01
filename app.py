from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = joblib.load('model/svm_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Extract features from the request
        features = [
            float(data['age']),
            float(data['daysPresence']),
            float(data['daysAbsence']),
            float(data['attendancePercentage']),
            float(data['englishBook']),
            float(data['bahasaMelayuBook']),
            float(data['mathTestMark']),
            float(data['mathTestMarkPercentage'])
        ]

        # Make a prediction using the model
        prediction = model.predict([features])[0]

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction)})

    except Exception as e:
        # Handle errors and return an error message
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)