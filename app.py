from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
features = joblib.load("model_features.pkl")   # ensure order

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract inputs in correct order
        data = []
        for col in features:
            value = request.form[col]
            data.append(float(value))

        final_input = np.array(data).reshape(1, -1)

        prediction = model.predict(final_input)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        return render_template("index.html",prediction_text=result,form_data=request.form)


    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
