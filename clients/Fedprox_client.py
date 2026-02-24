import copy
import torch
import torch.nn as nn
import torch.optim as optim

class FedProxClient:
    def __init__(self, model, trainloader, testloader, device='cpu', mu=0.001, epochs=1, lr=1e-4):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.mu = mu
        self.epochs = epochs
        self.lr = lr
        self.global_weights = copy.deepcopy(model.state_dict())

    def fit(self):
        """Train locally and return updated weights + average loss"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        n_samples = max(len(self.trainloader.dataset), 1)

        for _ in range(self.epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = loss_fn(outputs, y_batch)

                # FedProx proximal term
                prox_loss = 0.0
                for w, w0 in zip(self.model.parameters(), self.global_weights.values()):
                    prox_loss += ((w - w0.to(self.device))**2).sum()
                loss += (self.mu / 2) * prox_loss

                # Check NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: loss became NaN, skipping batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)  # smaller clip
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / n_samples
        return copy.deepcopy(self.model.state_dict()), avg_loss

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.global_weights = copy.deepcopy(state_dict)