# deploy_model.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("features")
    
    if not data:
        return jsonify({"error": "No features provided"}), 400

    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
