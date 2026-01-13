# client.py
import flwr as fl
import numpy as np
import tensorflow as tf
from model import create_model
import json
import sys

# Load data
X = np.load('X.npy')
y = np.load('y.npy')

# Get client ID from command line
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Load partition
with open('client_partitions.json', 'r') as f:
    partitions = json.load(f)
indices = partitions[f'client_{client_id}']

X_local = X[indices]
y_local = y[indices]

print(f"Client {client_id}: {len(X_local)} samples, {y_local.sum()} high-risk")

# Create model
model = create_model(X.shape[1])

class CFClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_local, y_local, epochs=1, batch_size=16, verbose=0)
        return model.get_weights(), len(X_local), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc, prec, rec = model.evaluate(X_local, y_local, verbose=0)
        return loss, len(X_local), {"accuracy": acc, "precision": prec, "recall": rec}

# Start client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CFClient())