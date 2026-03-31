"""FieldNetwork: self-building spatial network on GPU — batch optimized."""

import torch
import torch.nn.functional as F
from src.field.neuron_field import NeuronField
from src.field.spatial import (extract_patches, extract_patches_batch,
                                build_activation_map, build_activation_map_batch)
from src.field.readout import Readout


class FieldNetwork:
    """Self-building network with spatial receptive fields.

    Batch-optimized: F.unfold processes all images at once, neuron positions
    are shared so the gather pattern is reused across the batch.
    """

    def __init__(
        self,
        patch_size: int = 5,
        initial_stride: int = 2,
        n_features: int = 8,
        capacity_per_layer: int = 2048,
        split_threshold: float = 0.50,
        merge_threshold: float = 0.85,
        growth_interval: int = 10,
        stagnation_threshold: float = 0.30,
        depth_patience: int = 5,
        depth_decay: float = 0.99,
        max_depth: int = 8,
        max_neurons_per_position: int = 16,
        readout_lr: float = 0.01,
        lr: float = 0.04,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.initial_stride = initial_stride
        self.n_features = n_features
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.growth_interval = growth_interval
        self.stagnation_threshold = stagnation_threshold
        self.depth_patience = depth_patience
        self.depth_decay = depth_decay
        self.max_depth = max_depth
        self.max_neurons_per_position = max_neurons_per_position
        self.lr = lr
        self.capacity_per_layer = capacity_per_layer

        # Layer 0: 5x5 patches on raw pixels
        layer0 = NeuronField(
            patch_dim=patch_size * patch_size,
            capacity=capacity_per_layer,
            device=str(self.device),
        )
        self.layers: list[NeuronField] = [layer0]
        self.layer_grid_sizes: list[tuple[int, int]] = []

        self._init_layer0()

        # Readout (resized on first forward)
        self.readout = Readout(
            feature_dim=1, n_classes=10, lr=readout_lr, device=str(self.device),
        )
        self._readout_initialized = False

        # Precomputed gather indices per layer (invalidated on growth)
        self._gather_cache: dict[int, dict] = {}

        # Stats
        self.step = 0
        self.n_width_grows = 0
        self.n_depth_grows = 0

    def _init_layer0(self):
        grid_h = (28 - self.patch_size) // self.initial_stride + 1
        grid_w = grid_h
        layer = self.layers[0]
        positions_y = []
        positions_x = []
        weights = []
        for r in range(grid_h):
            for c in range(grid_w):
                positions_y.append(r * self.initial_stride)
                positions_x.append(c * self.initial_stride)
                weights.append(torch.randn(self.patch_size ** 2, device=self.device) * 0.1)

        W = torch.stack(weights)
        pos_x = torch.tensor(positions_x, dtype=torch.long, device=self.device)
        pos_y = torch.tensor(positions_y, dtype=torch.long, device=self.device)
        labels = torch.full((len(weights),), -1, dtype=torch.long, device=self.device)
        layer.add_neurons_batch(W, pos_x, pos_y, labels)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _get_gather_info(self, layer: NeuronField, layer_idx: int,
                         H_out: int, W_out: int) -> dict:
        """Get or compute cached gather indices for a layer."""
        cache_key = layer_idx
        cached = self._gather_cache.get(cache_key)
        if cached and cached.get('n_alive') == layer.n_alive:
            return cached

        alive_idx = layer.get_alive_indices()
        if len(alive_idx) == 0:
            info = {'alive_idx': alive_idx, 'n_alive': 0, 'patch_indices': None, 'positions': None}
            self._gather_cache[cache_key] = info
            return info

        alive_py = layer.pos_y[alive_idx].clamp(0, H_out - 1)
        alive_px = layer.pos_x[alive_idx].clamp(0, W_out - 1)
        patch_indices = (alive_py * W_out + alive_px).clamp(0, H_out * W_out - 1)
        positions = torch.stack([alive_py, alive_px], dim=1)

        info = {
            'alive_idx': alive_idx,
            'n_alive': layer.n_alive,
            'patch_indices': patch_indices,
            'positions': positions,
            'W_alive': layer.W[alive_idx],
        }
        self._gather_cache[cache_key] = info
        return info

    def _layer_forward_batch(self, layer: NeuronField, input_maps: torch.Tensor,
                             layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch forward through one layer.

        Args:
            layer: NeuronField
            input_maps: (B, C, H, W)
            layer_idx: int

        Returns:
            activations: (B, n_alive)
            output_maps: (B, 1, H_out, W_out)
        """
        B = input_maps.shape[0]
        H_in, W_in = input_maps.shape[2], input_maps.shape[3]
        H_out = H_in - self.patch_size + 1
        W_out = W_in - self.patch_size + 1

        if H_out <= 0 or W_out <= 0:
            empty = torch.zeros(B, 1, max(H_out, 1), max(W_out, 1), device=self.device)
            return torch.zeros(B, 0, device=self.device), empty

        while len(self.layer_grid_sizes) <= layer_idx:
            self.layer_grid_sizes.append((0, 0))
        self.layer_grid_sizes[layer_idx] = (H_out, W_out)

        # Check/fix patch dim
        expected_patch_dim = input_maps.shape[1] * self.patch_size * self.patch_size
        if layer.patch_dim != expected_patch_dim:
            new_layer = NeuronField(expected_patch_dim, self.capacity_per_layer, str(self.device))
            alive_idx = layer.get_alive_indices()
            if len(alive_idx) > 0:
                n = len(alive_idx)
                new_W = torch.randn(n, expected_patch_dim, device=self.device) * 0.1
                new_layer.add_neurons_batch(new_W, layer.pos_x[alive_idx],
                                           layer.pos_y[alive_idx], layer.labels[alive_idx])
            self.layers[layer_idx] = new_layer
            layer = new_layer
            self._gather_cache.pop(layer_idx, None)

        info = self._get_gather_info(layer, layer_idx, H_out, W_out)
        n_alive = info['n_alive']

        if n_alive == 0:
            empty = torch.zeros(B, 1, H_out, W_out, device=self.device)
            return torch.zeros(B, 0, device=self.device), empty

        # Batch extract patches: (B, n_positions, patch_dim)
        patches_all = extract_patches_batch(input_maps, self.patch_size, stride=1)

        # Gather patches for alive neurons: (B, n_alive, patch_dim)
        neuron_patches = patches_all[:, info['patch_indices'], :]

        # Batch cosine similarity: (B, n_alive)
        W_alive = info['W_alive']  # (n_alive, patch_dim)
        dots = (neuron_patches * W_alive.unsqueeze(0)).sum(dim=2)  # (B, n_alive)
        w_norms = W_alive.norm(dim=1).clamp(min=1e-9)  # (n_alive,)
        p_norms = neuron_patches.norm(dim=2).clamp(min=1e-9)  # (B, n_alive)
        sims = dots / (p_norms * w_norms.unsqueeze(0))  # (B, n_alive)
        acts = torch.relu(sims)  # (B, n_alive)

        # Lateral inhibition per image
        if n_alive > 1:
            k = max(1, int(n_alive * 0.3))
            if k < n_alive:
                topk_vals, _ = torch.topk(acts, k, dim=1)
                threshold = topk_vals[:, -1:]  # (B, 1)
                mask = acts >= threshold
                acts = torch.where(mask, acts, acts * 0.1)

        # Build batch activation maps: (B, 1, H_out, W_out)
        output_maps = build_activation_map_batch(info['positions'], acts, H_out, W_out)

        return acts, output_maps

    def _layer_forward(self, layer, input_map, layer_idx):
        """Single-image forward (for compatibility)."""
        acts, output_map = self._layer_forward_batch(layer, input_map, layer_idx)
        return acts[0] if acts.shape[0] > 0 else acts, output_map

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Image (1,1,28,28) -> flat feature vector."""
        current_map = image
        for i, layer in enumerate(self.layers):
            _, current_map = self._layer_forward_batch(layer, current_map, i)
        features = current_map.flatten()
        if not self._readout_initialized or features.shape[0] != self.readout.feature_dim:
            self.readout.resize(features.shape[0])
            self._readout_initialized = True
        return features

    def forward_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Batch forward: (B,1,28,28) -> (B, feature_dim)."""
        current_maps = images
        for i, layer in enumerate(self.layers):
            _, current_maps = self._layer_forward_batch(layer, current_maps, i)
        # Flatten spatial dims per image
        B = current_maps.shape[0]
        features = current_maps.view(B, -1)  # (B, n_features * H * W)
        if not self._readout_initialized or features.shape[1] != self.readout.feature_dim:
            self.readout.resize(features.shape[1])
            self._readout_initialized = True
        return features

    def predict(self, image: torch.Tensor) -> int:
        with torch.no_grad():
            features = self.forward(image)
            return self.readout.predict(features)

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Fully batched prediction."""
        with torch.no_grad():
            features = self.forward_batch(images)
            return self.readout.predict_batch(features)

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor,
                    batch_size: int = 256):
        """Batch training: batched forward + batched readout + per-batch Hebbian."""
        n = images.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]
            batch_labels = labels[start:end]
            bs = batch_imgs.shape[0]
            self.step += bs

            # === Batched forward through all spatial layers ===
            current_maps = batch_imgs
            layer_input_maps = []  # save inputs for Hebbian update

            for li, layer in enumerate(self.layers):
                layer_input_maps.append(current_maps)
                acts, current_maps = self._layer_forward_batch(layer, current_maps, li)

                # Hebbian update: for each image, find best neuron, update it
                info = self._get_gather_info(layer, li,
                    self.layer_grid_sizes[li][0], self.layer_grid_sizes[li][1])
                if info['n_alive'] > 0 and acts.shape[1] > 0:
                    best_local = acts.argmax(dim=1)  # (bs,)
                    best_sims = acts[torch.arange(bs, device=self.device), best_local]

                    merge_mask = best_sims >= self.merge_threshold
                    if merge_mask.any():
                        # Use the INPUT to this layer for patch extraction
                        patches_all = extract_patches_batch(
                            layer_input_maps[li], self.patch_size, stride=1)

                        m_indices = merge_mask.nonzero(as_tuple=True)[0]
                        for img_i in m_indices:
                            img_i = int(img_i)
                            local_best = int(best_local[img_i])
                            global_best = int(info['alive_idx'][local_best])
                            patch_idx = int(info['patch_indices'][local_best])
                            if patch_idx < patches_all.shape[1] and \
                               patches_all.shape[2] == layer.patch_dim:
                                patch = patches_all[img_i, patch_idx]
                                layer.W[global_best] += self.lr * (patch - layer.W[global_best])
                            layer.labels[global_best] = int(batch_labels[img_i])

                    self._gather_cache.pop(li, None)

            # === Batched readout gradient step ===
            B = current_maps.shape[0]
            features = current_maps.view(B, -1).detach()
            if not self._readout_initialized or features.shape[1] != self.readout.feature_dim:
                self.readout.resize(features.shape[1])
                self._readout_initialized = True
            self.readout.train_step(features, batch_labels[:B])

            # Growth check
            if self.step % (self.growth_interval * batch_size) < batch_size:
                self._check_growth_fast()

    def learn(self, image: torch.Tensor, label: int):
        """Single-image learn (wraps train_batch)."""
        self.train_batch(image.unsqueeze(0) if image.dim() == 3 else image,
                        torch.tensor([label], device=self.device), batch_size=1)

    def _check_growth_fast(self):
        """Add neurons at uncovered positions."""
        for i, layer in enumerate(self.layers):
            if i >= len(self.layer_grid_sizes):
                break
            H_out, W_out = self.layer_grid_sizes[i]
            if H_out == 0 or W_out == 0:
                continue
            alive = layer.n_alive
            n_positions = H_out * W_out

            if alive < n_positions and alive < layer.capacity:
                covered = set()
                alive_idx = layer.get_alive_indices()
                for j in alive_idx:
                    j = int(j)
                    covered.add((int(layer.pos_y[j]), int(layer.pos_x[j])))

                n_grow = min(5, n_positions - len(covered))
                grow_W = []
                grow_px = []
                grow_py = []
                for _ in range(n_grow):
                    py = torch.randint(0, max(H_out, 1), (1,)).item()
                    px = torch.randint(0, max(W_out, 1), (1,)).item()
                    if (py, px) not in covered:
                        grow_W.append(torch.randn(layer.patch_dim, device=self.device) * 0.1)
                        grow_px.append(px)
                        grow_py.append(py)
                        covered.add((py, px))
                        self.n_width_grows += 1

                if grow_W:
                    layer.add_neurons_batch(
                        torch.stack(grow_W),
                        torch.tensor(grow_px, dtype=torch.long, device=self.device),
                        torch.tensor(grow_py, dtype=torch.long, device=self.device),
                        torch.full((len(grow_W),), -1, dtype=torch.long, device=self.device),
                    )
                    self._gather_cache.pop(i, None)

    def total_neurons(self) -> int:
        return sum(l.n_alive for l in self.layers)

    def topology(self) -> dict:
        return {
            "layers": self.n_layers,
            "total_neurons": self.total_neurons(),
            "per_layer": [l.n_alive for l in self.layers],
            "grid_sizes": list(self.layer_grid_sizes),
            "n_width_grows": self.n_width_grows,
            "n_depth_grows": self.n_depth_grows,
        }
