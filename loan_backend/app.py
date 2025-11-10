from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
app = Flask(__name__)
CORS(app)   # ✅ This allows React to connect

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
    return jsonify({"prediction": result})

app.run(port=5000)
