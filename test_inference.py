# test_inference.py
import numpy as np

from ai_edge_litert.interpreter import Interpreter
interpreter = Interpreter(model_path="cf_screening_model.tflite")


# import tensorflow as tf
# interpreter = tf.lite.Interpreter(model_path="cf_screening_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Simulate a high-risk patient (based on your synthetic schema)
# Features: [age_months, family_history_cf, ethnicity, salty_skin, cough_type,
#            respiratory_infections_frequency, wheezing_present, weight_percentile,
#            growth_faltering, stool_character, meconium_ileus,
#            sweat_test_simulated, respiratory_score, nutritional_risk_score]

high_risk_sample = np.array([[
    6.0,      # age_months
    1.0,      # family_history_cf
    0.0,      # ethnicity (Caucasian = 0 after encoding)
    1.0,      # salty_skin
    2.0,      # cough_type
    3.0,      # respiratory_infections_frequency
    1.0,      # wheezing_present
    8.0,      # weight_percentile
    1.0,      # growth_faltering
    1.0,      # stool_character (e.g., 'greasy' = encoded as 1)
    1.0,      # meconium_ileus
    95.0,     # sweat_test_simulated
    7.0,      # respiratory_score
    8.0       # nutritional_risk_score
]], dtype=np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], high_risk_sample)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

risk_score = output[0][0]
print(f"Risk score: {risk_score:.4f}")
if risk_score >= 0.5:
    print("⚠️  HIGH RISK: Refer for sweat test!")
else:
    print("✅ Low risk: Routine care.")