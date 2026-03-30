"""Global readout layer: 10 neurons, gradient-trained."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout:
    """Linear readout from spatial feature map to class predictions.
    The only gradient-trained component.
    """

    def __init__(self, feature_dim: int, n_classes: int = 10, lr: float = 0.01,
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.lr = lr
        self.feature_dim = feature_dim
        self.linear = nn.Linear(feature_dim, n_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.to(self.device))

    def predict(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self(features)
            return int(logits.argmax(-1))

    def predict_batch(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self(features)
            return logits.argmax(dim=-1)

    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        logits = self(features.to(self.device))
        loss = F.cross_entropy(logits, labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def resize(self, new_feature_dim: int):
        if new_feature_dim == self.feature_dim:
            return
        old_weight = self.linear.weight.data
        self.feature_dim = new_feature_dim
        self.linear = nn.Linear(new_feature_dim, self.n_classes).to(self.device)
        min_dim = min(old_weight.shape[1], new_feature_dim)
        with torch.no_grad():
            self.linear.weight[:, :min_dim] = old_weight[:, :min_dim]
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.lr)
