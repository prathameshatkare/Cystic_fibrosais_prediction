# server.py
import flwr as fl
import numpy as np
import tensorflow as tf
from model import create_model

# Load feature dimension (from X.npy)
X = np.load('X.npy')
input_dim = X.shape[1]

def fit_config(server_round: int):
    return {"server_round": server_round}

def evaluate_config(server_round: int):
    return {"server_round": server_round}

# Strategy with initial parameters
initial_model = create_model(input_dim)
initial_weights = fl.common.ndarrays_to_parameters(initial_model.get_weights())

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=initial_weights,
)

# Start server and get final parameters
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

# ❌ Problem: We can't easily get final weights with start_server()

# ✅ Better approach: Use Simulation Engine (recommended for saving models)