"""NeuronField: pre-allocated per-layer tensor storage for spatial neurons."""

import torch


class NeuronField:
    """Pre-allocated tensor block for one layer's neurons.
    All neurons share the same patch_dim. Growth = write to dead slot or expand.
    """

    def __init__(self, patch_dim: int, capacity: int = 2048, device: str = "cuda"):
        self.patch_dim = patch_dim
        self.capacity = capacity
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.W = torch.zeros(capacity, patch_dim, device=self.device)
        self.pos_x = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.pos_y = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.labels = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.fire_counts = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.last_fire = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.ages = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.alive = torch.zeros(capacity, dtype=torch.bool, device=self.device)

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def add_neuron(self, w: torch.Tensor, pos_x: int, pos_y: int, label: int = -1):
        w = w.to(self.device)
        dead_slots = (~self.alive).nonzero(as_tuple=True)[0]
        if len(dead_slots) > 0:
            idx = int(dead_slots[0])
        else:
            self._expand()
            idx = int((~self.alive).nonzero(as_tuple=True)[0][0])
        self.W[idx] = w
        self.pos_x[idx] = pos_x
        self.pos_y[idx] = pos_y
        self.labels[idx] = label
        self.fire_counts[idx] = 0
        self.last_fire[idx] = 0
        self.ages[idx] = 0
        self.alive[idx] = True

    def add_neurons_batch(self, W: torch.Tensor, pos_x: torch.Tensor,
                          pos_y: torch.Tensor, labels: torch.Tensor):
        n = W.shape[0]
        dead_slots = (~self.alive).nonzero(as_tuple=True)[0]
        while len(dead_slots) < n:
            self._expand()
            dead_slots = (~self.alive).nonzero(as_tuple=True)[0]
        slots = dead_slots[:n]
        self.W[slots] = W.to(self.device)
        self.pos_x[slots] = pos_x.to(self.device)
        self.pos_y[slots] = pos_y.to(self.device)
        self.labels[slots] = labels.to(self.device)
        self.fire_counts[slots] = 0
        self.last_fire[slots] = 0
        self.ages[slots] = 0
        self.alive[slots] = True

    def _expand(self):
        old_cap = self.capacity
        new_cap = old_cap * 2
        for attr in ['W', 'pos_x', 'pos_y', 'labels', 'fire_counts', 'last_fire', 'ages', 'alive']:
            old = getattr(self, attr)
            if attr == 'W':
                new = torch.zeros(new_cap, self.patch_dim, device=self.device)
            elif attr == 'alive':
                new = torch.zeros(new_cap, dtype=torch.bool, device=self.device)
            elif attr == 'labels':
                new = torch.full((new_cap,), -1, dtype=torch.long, device=self.device)
            else:
                new = torch.zeros(new_cap, dtype=old.dtype, device=self.device)
            new[:old_cap] = old
            setattr(self, attr, new)
        self.capacity = new_cap

    def batch_cosine_sim(self, patches: torch.Tensor, neuron_indices: torch.Tensor) -> torch.Tensor:
        """Cosine sim between each neuron and its corresponding patch. (n,) output."""
        W_sel = self.W[neuron_indices]
        dot = (W_sel * patches).sum(dim=1)
        w_norm = W_sel.norm(dim=1).clamp(min=1e-9)
        p_norm = patches.norm(dim=1).clamp(min=1e-9)
        return dot / (w_norm * p_norm)

    def batch_cosine_sim_all(self, patches: torch.Tensor) -> torch.Tensor:
        """All alive neurons vs all patches: (n_alive, n_patches)."""
        alive_W = self.W[self.alive]
        w_norm = alive_W.norm(dim=1).clamp(min=1e-9)
        p_norm = patches.norm(dim=1).clamp(min=1e-9)
        dots = alive_W @ patches.T
        return dots / (w_norm.unsqueeze(1) * p_norm.unsqueeze(0))

    def prune(self, prune_age: int = 3000):
        dead = self.alive & (self.fire_counts == 0) & (self.ages > prune_age)
        self.alive[dead] = False

    def get_alive_mask(self) -> torch.Tensor:
        return self.alive.clone()

    def get_alive_indices(self) -> torch.Tensor:
        return self.alive.nonzero(as_tuple=True)[0]
