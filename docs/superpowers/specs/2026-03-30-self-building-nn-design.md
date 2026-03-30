# Self-Building Neural Network — Design Spec

**Date:** 2026-03-30
**Status:** Approved

---

## Problem

Every successful neural network architecture (YOLO, ResNet, Transformers) has its topology hand-designed by researchers. NEAT showed that topology can emerge through evolutionary search, but evolution is slow and disconnected from the learning signal. Biological brains grow structure in real-time — synaptogenesis driven by activity, pruning driven by disuse. No architect, no generations.

We want to build a framework that explores **self-building neural networks**: networks that start minimal and grow both width (new neurons) and depth (new layers) at runtime, guided by biological signals — not random mutation, not human design.

## Goals

1. A network that starts with fixed input/output and 2 small hidden layers, then grows its own topology during training
2. Pluggable neuron types — the growth engine is neuron-agnostic
3. Full benchmark suite (B1-B5) to evaluate any neuron type under the same growth rules
4. V1: prototype neuron + perceptron baseline, validated on MNIST

## Non-Goals (V1)

- Cross-type layers (mixing neuron types within a layer)
- Recurrent connections
- Convolutional structure (flat vectors only, MNIST is the testbed)
- Spiking, dendritic, or predictive coding neurons (future plugins)

---

## Architecture

### 1. Neuron (pluggable unit)

Abstract interface that all neuron types implement:

```
class Neuron:
    activate(input: ndarray) -> float        # forward pass, return activation
    update(input: ndarray, lr: float)        # learn from input
    similarity(input: ndarray) -> float      # how well does this neuron match the input
    copy() -> Neuron                         # duplicate self (for mitosis)
    state() -> dict                          # introspection for benchmarks
```

**V1 neuron types:**

- `PrototypeNeuron` — weight vector + cosine similarity + competitive absorption. This is the working neuron from the existing research, stripped of HVM/interaction-net framing.
- `PerceptronNeuron` — standard w*x+b with sigmoid/relu, gradient update. The baseline that every architecture uses. Included to measure what self-building topology gives you over a static network of perceptrons.

**Future neuron types** (not V1, but the interface must support them):
- LIF spiking neuron (spike timing, STDP learning)
- Dendritic neuron (multi-compartment, per-dendrite nonlinearity)
- Predictive coding neuron (top-down prediction, bottom-up error)

### 2. Layer

An ordered collection of neurons, all the same type.

**State tracked per layer:**
- `activations`: output vector from last forward pass
- `activation_variance`: variance across neurons — measures information throughput. Low variance = bottleneck (all neurons responding the same way).
- `reconstruction_error`: how well next layer's activations can reconstruct this layer's output. High = information is being lost or transformed (transformation is good; loss is bad).
- `neuron_fire_counts`: per-neuron usage tracking for pruning

**Operations:**
- `forward(input) -> activations` — activate all neurons, return vector
- `grow(n=1)` — add n neurons (duplicated from the highest-firing existing neuron + small noise)
- `prune()` — remove neurons that haven't fired in T steps, respecting minimum layer size
- `duplicate() -> Layer` — copy the entire layer (for depth growth). All neurons are copied via `neuron.copy()` with small noise added to break symmetry.

### 3. Growth Engine (the core contribution)

The network starts as: `Input(784) -> H1(k) -> H2(k) -> Output(10)` where k is small (e.g., 8-16 neurons).

Every `growth_interval` steps, the engine evaluates two signals:

#### Signal 1: Error (width growth)

Per-layer, local signal. After a forward pass:

```
For each layer L:
    best_sim = max similarity between input and any neuron in L
    if prediction was wrong AND best_sim < merge_threshold:
        -> L needs more coverage -> grow L (add neuron)
```

This is the existing "DUP rule" generalized to work per-layer instead of in a flat bag. A neuron is added when the layer doesn't have a good match for the input pattern — the same logic that currently creates new HVM nodes.

**Width growth mechanism:** Duplicate the most active neuron in the layer, add small Gaussian noise to break symmetry. The duplicate will initially respond to similar patterns, then specialize as learning diverges them.

#### Signal 2: Information (depth growth)

Between-layer, global signal. Measured every `depth_check_interval` steps (less frequently than width checks):

```
For each adjacent pair (L_a, L_b):
    correlation = correlation between L_a.activations and L_b.activations
    if correlation > depth_threshold (representations too similar):
        -> this pair isn't transforming enough -> duplicate L_a between them
```

High correlation between adjacent layers means the second layer is mostly passing through what the first layer computed — it's not adding representational power. Inserting a duplicate gives the network more capacity to learn a transformation there.

**Depth growth mechanism (mitosis):**
1. Duplicate layer L_a to create L_a'
2. Insert L_a' between L_a and L_b
3. Add small noise to L_a' neuron weights to break symmetry
4. L_a and L_a' now receive different error signals and will specialize over time

If the duplication was unnecessary, the prune mechanism will collapse it — neurons in the redundant layer stop being the best match for any input, stop firing, and get pruned. If all neurons in a layer are pruned, the layer itself is removed.

**Key insight:** Width and depth growth are the same mechanism at different scales. Width = duplicate a neuron. Depth = duplicate a layer. Prune = remove unused neurons/layers. The brain does this with synaptogenesis, dendritic branching, and synaptic pruning.

#### Signal 3: Prune (structural cleanup)

```
For each neuron n in each layer L:
    if n.fire_count == 0 and n.age > prune_age:
        remove n from L
    if n.last_fire < (step - prune_window) and n.fire_count > 0:
        remove n from L

For each layer L:
    if L.size == 0:
        remove L from network
    if L.size < min_layer_size:
        protect from further pruning (minimum viable layer)
```

**Zero dead neuron guarantee:** Same as before — neurons are born from duplication (they initially match what their parent matched), and removed if they stop firing. Every living neuron has participated in at least one activation.

### 4. Infrastructure Modules

These are kept from the existing research. They work with any network/neuron type:

**DA Selection** (`da_selection.py`):
- `da_select(candidates) -> best, scores, index`
- The proven theorem: cosine_sim(z, z_bar) selects the MLE prototype
- Used during few-shot learning to pick the best example to commit

**Sleep Consolidation** (`sleep.py`):
- Hippocampal buffer (fast, episodic) + network replay (slow, semantic)
- Interleaved replay prevents catastrophic forgetting
- After sleep: trigger prune cycle, tighten growth thresholds (ACh decay)

**Theta Buffer** (`theta.py`):
- 7-slot circular buffer for sequence encoding
- Bigram transition tracking
- Sequence completion from prefix

### 5. Benchmark Harness

All five benchmarks, neuron-agnostic:

| Benchmark | Tests | Metric |
|-----------|-------|--------|
| B1: Classification | Full MNIST 60k/10k | Accuracy, node count, dead neurons |
| B2: DA Theorem | Prototype selection vs K | DA gain over random, oracle gap, O(1/sqrt(K)) scaling |
| B3: Split MNIST | 5 tasks x 2 classes, continual | Per-task accuracy, BWT (backward transfer) |
| B4: Combined | One-shot + retention + sequence | Overall accuracy, day retention, seq completion |
| B5: Ablation | Toggle each component | Marginal contribution of DA, sleep, theta, growth |

The harness takes a `Network` instance and runs all benchmarks, producing a standardized report. This lets you swap neuron types and compare directly.

---

## Repo Structure

```
self-building-nn/
    src/
        neurons/
            base.py             # Abstract Neuron interface
            prototype.py        # PrototypeNeuron (competitive, cosine sim)
            perceptron.py       # PerceptronNeuron (standard, gradient)
        network/
            layer.py            # Layer (collection of neurons, stats tracking)
            network.py          # Network + GrowthEngine (width/depth/prune)
        signals/
            error.py            # Error signal (per-layer, width trigger)
            information.py      # Information signal (between-layer, depth trigger)
        infrastructure/
            da_selection.py     # DA prototype selection (theorem)
            sleep.py            # Sleep consolidation (CLS + interleaved replay)
            theta.py            # Theta-gamma temporal buffer
        benchmarks/
            harness.py          # Run all benchmarks on any Network
            b1_classification.py
            b2_da_theorem.py
            b3_split_mnist.py
            b4_combined.py
            b5_ablation.py
        data/
            mnist.py            # MNIST download + loader
    tests/
        test_neurons.py
        test_network.py
        test_growth.py
        test_benchmarks.py
    docs/
        superpowers/
            specs/
                2026-03-30-self-building-nn-design.md
    README.md
    requirements.txt            # numpy, scikit-learn, scipy
```

---

## Growth Parameters (V1 defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_hidden_size` | 16 | Neurons per hidden layer at start |
| `growth_interval` | 50 | Check width growth every N steps |
| `depth_check_interval` | 500 | Check depth growth every N steps |
| `merge_threshold` | 0.80 | Similarity above which neuron absorbs input |
| `split_threshold` | 0.35 | Similarity below which layer needs more coverage |
| `depth_threshold` | 0.90 | Activation correlation above which layers are too similar |
| `prune_age` | 3000 | Steps before an unfired neuron is pruned |
| `prune_window` | 3000 | Steps since last fire before active neuron is pruned |
| `min_layer_size` | 4 | Minimum neurons per layer. If a layer is at min size and all neurons are inactive, the entire layer is removed. |
| `growth_noise` | 0.005 | Gaussian noise added to duplicated neurons |
| `ach_decay` | 0.55 | ACh decay rate per sleep cycle (tightens thresholds) |
| `max_layers` | 20 | Safety cap on depth growth |
| `max_neurons_per_layer` | 512 | Safety cap on width growth |

---

## Success Criteria

V1 is successful if:

1. The network self-sizes from 2 hidden layers to a topology that outperforms the fixed-size 2-layer baseline on B1
2. Zero dead neurons maintained at all scales
3. Prototype neuron beats perceptron baseline on B3 (continual learning) and B4 (combined task)
4. Growth dynamics are observable — you can see when/why the network adds neurons and layers
5. Adding a new neuron type requires implementing only the Neuron interface (< 50 lines)
