from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = joblib.load('model/svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
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
        
        prediction = model.predict([features])[0]
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)