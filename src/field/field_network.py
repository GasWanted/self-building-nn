"""FieldNetwork: self-building spatial network on GPU."""

import torch
import torch.nn.functional as F
from src.field.neuron_field import NeuronField
from src.field.spatial import extract_patches, build_activation_map
from src.field.readout import Readout


class FieldNetwork:
    """Self-building network with spatial receptive fields.

    Processes raw 28x28 images. Each layer's neurons have local
    receptive fields. Growth adds neurons at positions with poor coverage.
    Depth growth duplicates the deepest layer.
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

        # Layer 0: 5x5 patches on raw pixels, patch_dim = 25
        layer0 = NeuronField(
            patch_dim=patch_size * patch_size,
            capacity=capacity_per_layer,
            device=str(self.device),
        )
        self.layers: list[NeuronField] = [layer0]
        self.layer_grid_sizes: list[tuple[int, int]] = []

        # Initialize layer 0 neurons at stride intervals
        self._init_layer0()

        # Readout (resized on first forward)
        self.readout = Readout(
            feature_dim=1, n_classes=10, lr=readout_lr, device=str(self.device),
        )
        self._readout_initialized = False

        # Stats
        self.step = 0
        self.n_width_grows = 0
        self.n_depth_grows = 0
        self.layer_wrong_counts: list[float] = [0.0]

    def _init_layer0(self):
        """Place initial neurons on the 28x28 pixel grid at stride intervals."""
        grid_h = (28 - self.patch_size) // self.initial_stride + 1
        grid_w = grid_h
        layer = self.layers[0]
        positions_y = []
        positions_x = []
        weights = []
        for r in range(grid_h):
            for c in range(grid_w):
                py = r * self.initial_stride
                px = c * self.initial_stride
                positions_y.append(py)
                positions_x.append(px)
                weights.append(torch.randn(self.patch_size ** 2, device=self.device) * 0.1)

        W = torch.stack(weights)
        pos_x = torch.tensor(positions_x, dtype=torch.long, device=self.device)
        pos_y = torch.tensor(positions_y, dtype=torch.long, device=self.device)
        labels = torch.full((len(weights),), -1, dtype=torch.long, device=self.device)
        layer.add_neurons_batch(W, pos_x, pos_y, labels)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _layer_forward(self, layer: NeuronField, input_map: torch.Tensor,
                       layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward one layer: input_map -> activations + output_map.

        Args:
            layer: NeuronField for this layer
            input_map: (1, C, H, W)
            layer_idx: index of this layer

        Returns:
            activations: (n_alive,) per-neuron
            output_map: (1, n_features, H_out, W_out)
        """
        H_in, W_in = input_map.shape[2], input_map.shape[3]
        H_out = H_in - self.patch_size + 1
        W_out = W_in - self.patch_size + 1

        if H_out <= 0 or W_out <= 0:
            empty = torch.zeros(1, self.n_features, max(H_out, 1), max(W_out, 1),
                              device=self.device)
            return torch.tensor([], device=self.device), empty

        # Store grid size
        while len(self.layer_grid_sizes) <= layer_idx:
            self.layer_grid_sizes.append((0, 0))
        self.layer_grid_sizes[layer_idx] = (H_out, W_out)

        # Extract all patches
        patches = extract_patches(input_map, self.patch_size, stride=1)
        # patches: (H_out * W_out, patch_dim)

        # Check patch dim matches layer
        expected_patch_dim = patches.shape[1]
        if layer.patch_dim != expected_patch_dim:
            # Rebuild layer with correct patch dim (happens when deeper layers see multi-channel input)
            new_layer = NeuronField(expected_patch_dim, self.capacity_per_layer, str(self.device))
            # Re-initialize with random weights at existing positions
            alive_idx = layer.get_alive_indices()
            if len(alive_idx) > 0:
                n = len(alive_idx)
                new_W = torch.randn(n, expected_patch_dim, device=self.device) * 0.1
                new_layer.add_neurons_batch(
                    new_W,
                    layer.pos_x[alive_idx],
                    layer.pos_y[alive_idx],
                    layer.labels[alive_idx],
                )
            self.layers[layer_idx] = new_layer
            layer = new_layer

        alive_idx = layer.get_alive_indices()
        if len(alive_idx) == 0:
            empty = torch.zeros(1, self.n_features, H_out, W_out, device=self.device)
            return torch.tensor([], device=self.device), empty

        # Get each neuron's patch based on position
        alive_py = layer.pos_y[alive_idx].clamp(0, H_out - 1)
        alive_px = layer.pos_x[alive_idx].clamp(0, W_out - 1)
        patch_indices = alive_py * W_out + alive_px
        patch_indices = patch_indices.clamp(0, patches.shape[0] - 1)
        neuron_patches = patches[patch_indices]  # (n_alive, patch_dim)

        # Batch cosine similarity
        sims = layer.batch_cosine_sim(neuron_patches, alive_idx)
        activations = torch.relu(sims)

        # Lateral inhibition (global top fraction for speed)
        n_alive = len(alive_idx)
        if n_alive > 1:
            k = max(1, int(n_alive * 0.3))
            if k < n_alive:
                topk_vals, _ = torch.topk(activations, k)
                threshold = topk_vals[-1]
                mask = activations >= threshold
                activations = torch.where(mask, activations, activations * 0.1)

        # Track firing
        layer.ages[alive_idx] += 1
        fired = activations > 0
        layer.fire_counts[alive_idx] += fired.long()
        step_tensor = torch.tensor(self.step, device=self.device)
        layer.last_fire[alive_idx] = torch.where(fired, step_tensor, layer.last_fire[alive_idx])

        # Build output activation map
        positions = torch.stack([alive_py, alive_px], dim=1)
        output_map = build_activation_map(positions, activations, self.n_features, H_out, W_out)

        return activations, output_map

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Image (1,1,28,28) -> flat feature vector for readout."""
        current_map = image
        for i, layer in enumerate(self.layers):
            _, current_map = self._layer_forward(layer, current_map, i)

        features = current_map.flatten()

        if not self._readout_initialized or features.shape[0] != self.readout.feature_dim:
            self.readout.resize(features.shape[0])
            self._readout_initialized = True

        return features

    def predict(self, image: torch.Tensor) -> int:
        with torch.no_grad():
            features = self.forward(image)
            return self.readout.predict(features)

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        preds = []
        with torch.no_grad():
            for i in range(images.shape[0]):
                preds.append(self.predict(images[i:i+1]))
        return torch.tensor(preds, device=self.device)

    def learn(self, image: torch.Tensor, label: int):
        """One step: forward through spatial layers, Hebbian update, gradient readout."""
        self.step += 1

        current_map = image
        for i, layer in enumerate(self.layers):
            activations, current_map = self._layer_forward(layer, current_map, i)

            # Hebbian update on best-matching neuron
            alive_idx = layer.get_alive_indices()
            if len(alive_idx) > 0 and len(activations) > 0:
                best_local = int(activations.argmax())
                best_global = int(alive_idx[best_local])
                best_sim = float(activations[best_local])

                if best_sim >= self.merge_threshold:
                    H_out, W_out = self.layer_grid_sizes[i]
                    py = int(layer.pos_y[best_global].clamp(0, H_out - 1))
                    px = int(layer.pos_x[best_global].clamp(0, W_out - 1))
                    # Re-extract the patch this neuron sees
                    # For simplicity, use the stored weights + learning toward activation direction
                    layer.W[best_global] += self.lr * (
                        layer.W[best_global] * best_sim - layer.W[best_global]
                    ) if best_sim < 1.0 else torch.zeros_like(layer.W[best_global])
                    layer.labels[best_global] = label

        # Gradient readout training
        features = current_map.flatten().detach()
        if not self._readout_initialized or features.shape[0] != self.readout.feature_dim:
            self.readout.resize(features.shape[0])
            self._readout_initialized = True

        self.readout.train_step(features.unsqueeze(0),
                                torch.tensor([label], device=self.device))

        # Growth check
        if self.step % self.growth_interval == 0:
            self._check_growth_fast()

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor,
                    batch_size: int = 64):
        """Train on batch: Hebbian spatial updates + gradient readout."""
        n = images.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]
            batch_labels = labels[start:end]
            bs = batch_imgs.shape[0]

            features_list = []
            for i in range(bs):
                self.step += 1
                img = batch_imgs[i:i+1]
                lab = int(batch_labels[i])

                current_map = img
                for li, layer in enumerate(self.layers):
                    activations, current_map = self._layer_forward(layer, current_map, li)

                    # Hebbian update
                    alive_idx = layer.get_alive_indices()
                    if len(alive_idx) > 0 and len(activations) > 0:
                        best_local = int(activations.argmax())
                        best_global = int(alive_idx[best_local])
                        best_sim = float(activations[best_local])
                        if best_sim >= self.merge_threshold and li < len(self.layer_grid_sizes):
                            H_out, W_out = self.layer_grid_sizes[li]
                            patches = extract_patches(img if li == 0 else current_map,
                                                     self.patch_size, stride=1)
                            py = int(layer.pos_y[best_global].clamp(0, max(H_out-1, 0)))
                            px = int(layer.pos_x[best_global].clamp(0, max(W_out-1, 0)))
                            pidx = min(py * max(W_out, 1) + px, patches.shape[0] - 1)
                            if patches.shape[1] == layer.patch_dim:
                                patch = patches[pidx]
                                layer.W[best_global] += self.lr * (patch - layer.W[best_global])
                            layer.labels[best_global] = lab

                feat = current_map.flatten().detach()
                if not self._readout_initialized or feat.shape[0] != self.readout.feature_dim:
                    self.readout.resize(feat.shape[0])
                    self._readout_initialized = True
                features_list.append(feat)

            # Batch readout gradient step
            if features_list:
                feat_batch = torch.stack(features_list)
                self.readout.train_step(feat_batch, batch_labels[:len(features_list)])

            # Growth check once per mini-batch
            if self.step % self.growth_interval < batch_size:
                self._check_growth_fast()

    def _check_growth_fast(self):
        """Add neurons at positions with poor coverage."""
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
                for _ in range(n_grow):
                    py = torch.randint(0, max(H_out, 1), (1,)).item()
                    px = torch.randint(0, max(W_out, 1), (1,)).item()
                    if (py, px) not in covered:
                        w = torch.randn(layer.patch_dim, device=self.device) * 0.1
                        layer.add_neuron(w, pos_x=px, pos_y=py, label=-1)
                        covered.add((py, px))
                        self.n_width_grows += 1

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
