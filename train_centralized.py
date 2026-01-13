# train_centralized.py
import numpy as np
import tensorflow as tf
from model import create_model

# Load data
X = np.load('X.npy')
y = np.load('y.npy')

# Create and train model
model = create_model(X.shape[1])
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
loss, acc, prec, rec = model.evaluate(X, y, verbose=0)
print(f"\nFinal - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# Save model
model.save('cf_screening_model.h5')
print("✅ Model saved as 'cf_screening_model.h5'")