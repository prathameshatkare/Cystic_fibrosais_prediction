# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf  # Use TF Lite directly

app = Flask(__name__)

# Load TFLite model once at startup
interpreter = tf.lite.Interpreter(model_path="cf_screening_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract and validate inputs
        age = float(data['age'])
        family_hx = int(data['family_history'])
        salty_skin = int(data['salty_skin'])
        weight_pct = float(data['weight_percentile'])
        sweat_cl = float(data['sweat_chloride']) if data['sweat_chloride'] != '' else 25.0
        ethnicity = int(data['ethnicity'])

        # Default values for uncollected features
        cough_type = 0
        resp_infections = 0
        wheezing = 0
        stool_char = 0
        meconium_ileus = 0
        if age <= 2:
            meconium_ileus = int(data.get('meconium_ileus', 0))

        growth_faltering = 1 if weight_pct < 10 else 0
        resp_score = cough_type + resp_infections + wheezing
        nutr_score = (growth_faltering * 3) + (1 if stool_char != 0 else 0)

        # Prepare input array (must match training order!)
        sample = np.array([[
            age, family_hx, ethnicity, salty_skin,
            cough_type, resp_infections, wheezing, weight_pct,
            growth_faltering, stool_char, meconium_ileus,
            sweat_cl, resp_score, nutr_score
        ]], dtype=np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        risk_score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

        result = {
            "risk_score": round(risk_score, 4),
            "recommendation": "🚨 HIGH RISK: Refer for sweat test!" if risk_score >= 0.5 else "✅ LOW RISK: Routine care."
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)