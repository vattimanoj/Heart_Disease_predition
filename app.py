from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaled.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURE_ORDER = ["age", "sex", "cp", "thalach", "oldpeak", "slope", "thal"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:

            features = [float(request.form[col]) for col in FEATURE_ORDER]

            features_array = np.array([features])
            features_scaled = scaler.transform(features_array)

            prediction_value = model.predict(features_scaled)[0]
            prediction = "Heart Disease Detected" if prediction_value == 1 else "No Heart Disease"

            return render_template("index.html", prediction=prediction)

        except Exception as e:
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)