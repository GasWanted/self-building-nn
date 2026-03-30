# self-building-nn

Neural networks that build their own architecture at runtime.

No hand-designed topology. No evolutionary generations. The network starts with 2 small hidden layers and grows width (new neurons) and depth (new layers) driven by two biological signals:

- **Error signal**: layer lacks coverage for an input pattern -> add a neuron (width)
- **Information signal**: adjacent layers are too correlated -> duplicate a layer (depth)
- **Prediction-error depth signal**: layer is stagnant on wrong predictions -> duplicate (depth)
- **Prune**: neurons that stop firing get removed. Empty layers get removed.

Width and depth growth are the same mechanism at different scales: **mitosis**. Duplicate a neuron, duplicate a layer. The copies diverge through learning.

## Neuron types

The growth engine is neuron-agnostic. Swap the neuron type and rerun benchmarks.

| Type | Description | Status |
|------|-------------|--------|
| `PrototypeNeuron` | Competitive learning, cosine similarity | V2.1 |
| `PerceptronNeuron` | Standard w*x+b, gradient update | V2.1 |
| `DendriticNeuron` | Starts with 1 dendrite, grows on demand | V2.1 |
| `PredictiveCodingNeuron` | Cosine-similarity prediction, bottom-up error | V2.1 |
| `SpikingNeuron` | LIF with graded activation (v/v_thresh) | V2.1 |

## V2.1 changes

- **Dendritic**: starts with 1 dendrite, grows on demand (was: 4 random dendrites)
- **Predictive coding**: cosine similarity matching (was: L2-based)
- **Spiking**: graded activation `v/v_thresh` (was: binary spike)
- **Depth growth**: now triggers reliably (`stagnation_threshold=0.30`, `patience=5`, `max_layers=10`)
- **Lateral inhibition**: top-k winners, 30% winner fraction
- **Skip connections**: non-adjacent layer communication via `ConnectionTracker`

## V2 infrastructure

Modules that work with any network/neuron type:

- **Same-space propagation** — iterative refinement keeps all layers in the same vector space (no dimension change between layers).
- **Lateral inhibition** — top-k winners suppress others; sharpens competition between neurons.
- **Prediction-error depth signal** — stagnant layers on wrong predictions trigger depth growth.
- **Skip connections** — non-adjacent layer communication via `ConnectionTracker`. Auto-discovered and auto-pruned.
- **DA Selection** — proven MLE prototype selector: `cosine_sim(z, z_bar)`. Oracle gap O(1/sqrt(K)).
- **Sleep Consolidation** — hippocampal buffer + interleaved replay. Prevents catastrophic forgetting.
- **Theta Buffer** — sequence encoding via bigram co-occurrence. No positional embeddings.

## Benchmark results (V2.1, sklearn digits, PCA-50)

Results from `compare.py` — all 5 neuron types start at `[50, 8, 8, 10]` and self-build.

### Growth dynamics

| Neuron type | Initial topology | Final topology (B1) | Layers | Width grows |
|-------------|-----------------|---------------------|--------|-------------|
| Prototype   | [50, 8, 8, 10]  | [50, 348, 334, ..., 93, 10] | 12 | ~1700 |
| Perceptron  | [50, 8, 8, 10]  | [50, 341, 321, ..., 118, 10] | 12 | ~1850 |
| Dendritic   | [50, 8, 8, 10]  | [50, 341, 311, ..., 73, 10] | 12 | ~1400 |
| Predictive  | [50, 8, 8, 10]  | [50, 343, 321, ..., 136, 10] | 12 | ~1920 |
| Spiking     | [50, 8, 8, 10]  | [50, 512, 512, ..., 288, 10] | 12 | ~4160 |

All neuron types now grow to **12 layers** (from 4 initial) via depth growth (`stagnation_threshold=0.30`, `patience=5`, `max_layers=10`). Width growth fills each layer from 8 to 100-512 neurons driven by the error signal (split threshold = 0.50). Spiking neurons grow to the maximum (512) because their graded activations still produce low similarity scores.

### Head-to-head

| Benchmark | Metric | Prototype | Perceptron | Dendritic | Predictive | Spiking |
|-----------|--------|-----------|------------|-----------|------------|---------|
| B1 Classification | Test accuracy | **87.8%** | 85.6% | 46.9% | 86.7% | 1.7% |
| B2 DA Theorem | K=5 gain | +1.4% | +1.4% | +1.4% | +1.4% | +1.4% |
| B3 Split MNIST | BWT (forgetting) | **+7.3%** | +5.1% | -1.4% | +6.3% | +0.4% |
| B4 Combined | Overall accuracy | **34%** | 33% | 25% | **34%** | 7% |
| B4 Combined | Sequence completion | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

### B1: Classification

Full training on all 10 digit classes, with sleep consolidation replay.

- **Prototype**: 87.8% accuracy, 12 layers, 0 dead neurons
- **Perceptron**: 85.6% accuracy, 12 layers, 0 dead neurons
- **Predictive**: 86.7% accuracy, 12 layers, 0 dead neurons — now competitive (was 23.1% in V2)
- **Dendritic**: 46.9% accuracy, 12 layers, 0 dead neurons — improved from 49.4% with cleaner dendrite growth
- **Spiking**: 1.7% accuracy, 12 layers, 0 dead neurons — graded activation helps slightly (was 0.8%)

Prototype leads on classification. Predictive coding is now competitive thanks to cosine-similarity matching. Dendritic neurons capture patterns through adaptive dendrite growth. Spiking neurons remain limited by their activation dynamics on static image data.

### B3: Continual learning (Split MNIST)

5 sequential tasks ([0,1], [2,3], [4,5], [6,7], [8,9]) with sleep between tasks.

- **Prototype**: avg 66.2%, BWT = +7.3% (positive transfer — depth growth + sleep consolidation)
- **Predictive**: avg 65.2%, BWT = +6.3% (positive transfer — cosine prediction prevents overwriting)
- **Perceptron**: avg 55.8%, BWT = +5.1%
- **Dendritic**: avg 29.1%, BWT = -1.4% (mild forgetting)
- **Spiking**: avg 2.7%, BWT = +0.4% (no forgetting, but near-chance accuracy)

### B4: Combined (one-shot + retention + sequence)

3 learning days with DA-selected prototypes and theta-buffer sequence completion.

- All 5 neuron types achieve **5/5** sequence completions via the theta buffer
- Prototype and predictive lead on combined retention accuracy (34%), followed by perceptron (33%)

### Benchmark definitions

| # | Test | Measures |
|---|------|----------|
| B1 | Full classification | Accuracy, topology, dead neurons |
| B2 | DA theorem verification | Prototype selection gain vs K |
| B3 | Split MNIST (5 tasks) | Backward transfer (forgetting) |
| B4 | Combined task | One-shot + retention + sequence |
| B5 | Ablation | Marginal contribution of each component |

## Quick start

```bash
pip install -r requirements.txt
pytest tests/                                    # 85 tests
PYTHONPATH=. python3 run.py                      # prototype on sklearn digits
PYTHONPATH=. python3 run.py perceptron           # perceptron on sklearn digits
PYTHONPATH=. python3 run.py dendritic            # dendritic on sklearn digits
PYTHONPATH=. python3 run.py predictive           # predictive coding on sklearn digits
PYTHONPATH=. python3 run.py spiking              # spiking on sklearn digits
PYTHONPATH=. python3 run.py --data=mnist         # prototype on full MNIST (28x28)
PYTHONPATH=. python3 compare.py                  # all 5 neuron types head-to-head
PYTHONPATH=. python3 compare.py --data=mnist     # full MNIST comparison
```

## Structure

```
src/
  neurons/          # pluggable neuron types (5 types)
  network/          # layer + self-building network (growth engine)
  signals/          # error (width) + information (depth) triggers
  infrastructure/   # DA selection, sleep, theta buffer
  benchmarks/       # B1-B5 + harness
  data/             # MNIST loader (sklearn digits + full MNIST)
tests/
docs/superpowers/specs/
```
