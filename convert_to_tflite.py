# convert_to_tflite.py
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('cf_screening_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite file
with open('cf_screening_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TensorFlow Lite model saved as 'cf_screening_model.tflite'")