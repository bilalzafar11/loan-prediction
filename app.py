import sys
sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import joblib
import os

# ==========================
# Initialize Flask App
# ==========================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key")  # Production ready

# ==========================
# Load Model & Encoders
# ==========================
MODEL_PATH = "loan_model.pkl"
ENCODERS_PATH = "loan_encoders.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model, encoders = None, None

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    """Render home page with loan prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle loan prediction requests."""
    if not model or not encoders:
        flash("Model or encoders not loaded properly.", "danger")
        return redirect(url_for("home"))

    try:
        # Collect form data
        form_fields = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                       "Loan_Amount_Term", "Credit_History",
                       "Property_Area", "Education", "Self_Employed"]

        input_data = {}
        for field in form_fields:
            value = request.form.get(field)
            if value is None or value.strip() == "":
                flash(f"Missing value for {field}", "danger")
                return redirect(url_for("home"))
            input_data[field] = float(value) if field not in ["Property_Area", "Education", "Self_Employed"] else value

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply encoders for categorical features
        for col in ["Property_Area", "Education", "Self_Employed"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Predict
        prediction = model.predict(input_df)[0]

        # Flash result
        message = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        category = "success" if prediction == 1 else "danger"
        flash(message, category)

    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "danger")

    return redirect(url_for("home"))


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True") == "True"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
