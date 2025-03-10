from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('model/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature order (update according to your selected features)
FEATURES = ['specobjid', 'g', 'r', 'modelFlux_i', 'modelFlux_z', 'petroRad_g',
            'petroRad_r', 'petroFlux_i', 'petroFlux_z', 'psfMag_u']

# Define class mapping (update based on your dataset)
CLASS_MAPPING = {0: "STARBURST", 1:"STARFORMING"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        features = [float(request.form.get(name)) for name in FEATURES]
        features_array = np.array([features])  # Convert to numpy array

        # Predict the class
        prediction_class = model.predict(features_array)[0]  # Get class label
        class_name = CLASS_MAPPING.get(prediction_class, "Unknown")

        # Predict probabilities
        probabilities = model.predict_proba(features_array)[0]  # Get probability distribution

        # Get the highest probability
        prediction_probability = max(probabilities)  # Highest probability of the predicted class

        # Redirect to result page with prediction and probability
        return redirect(url_for('result', prediction=class_name, probability=prediction_probability))

    except Exception as e:
        return f"Error: {str(e)}", 400


@app.route('/result')
def result():
    prediction = request.args.get('prediction', "Unknown")  # Default to "Unknown" if missing
    probability = request.args.get('probability', "N/A")  # Default to "N/A" if missing

    try:
        probability = float(probability)  # Convert to float if possible
    except ValueError:
        probability = None  # Set to None if conversion fails

    return render_template('result.html', prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)