"""Global readout: gradient-trained MLP from feature map to classes."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout:
    """2-layer MLP readout from spatial feature map to class predictions.
    The only gradient-trained component.
    """

    def __init__(self, feature_dim: int, n_classes: int = 10, hidden: int = 128,
                 lr: float = 0.01, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.hidden = hidden
        self.lr = lr
        self.feature_dim = feature_dim
        self._build(feature_dim)

    def _build(self, feature_dim: int):
        self.net = nn.Sequential(
            nn.Linear(feature_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden, self.n_classes),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.to(self.device))

    def predict(self, features: torch.Tensor) -> int:
        self.net.eval()
        with torch.no_grad():
            logits = self(features)
        self.net.train()
        return int(logits.argmax(-1))

    def predict_batch(self, features: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            logits = self(features)
        self.net.train()
        return logits.argmax(dim=-1)

    def train_step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        self.net.train()
        self.optimizer.zero_grad()
        logits = self(features.to(self.device))
        loss = F.cross_entropy(logits, labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def resize(self, new_feature_dim: int):
        if new_feature_dim == self.feature_dim:
            return
        self.feature_dim = new_feature_dim
        self._build(new_feature_dim)
