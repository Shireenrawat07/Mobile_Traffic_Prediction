import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------------
# Define a simple model
# -------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# Model utils
# -------------------------------
def get_model_weights(model):
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]

def set_model_weights(model, weights):
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = torch.tensor(weights[i])
    model.load_state_dict(state_dict)

# -------------------------------
# Flower client
# -------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # Dummy data
        self.x_train = torch.randn(64, 32)
        self.y_train = torch.randn(64, 1)
        self.x_val = torch.randn(16, 32)
        self.y_val = torch.randn(16, 1)

    def get_parameters(self, config=None):
        return get_model_weights(self.model)

    def fit(self, parameters, config):
        set_model_weights(self.model, parameters)
        self.model.train()
        # One epoch for demo
        self.optimizer.zero_grad()
        y_pred = self.model(self.x_train)
        loss = self.criterion(y_pred, self.y_train)
        loss.backward()
        self.optimizer.step()
        return get_model_weights(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_model_weights(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.x_val)
            loss = self.criterion(y_pred, self.y_val)
        return float(loss.item()), len(self.x_val), {}

# -------------------------------
# Start client
# -------------------------------
if __name__ == "__main__":
    model = SimpleNN()
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(model)
    )
