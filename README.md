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
| `PrototypeNeuron` | Competitive learning, cosine similarity | V2 |
| `PerceptronNeuron` | Standard w*x+b, gradient update | V2 |
| `DendriticNeuron` | Multi-compartment, per-dendrite nonlinearity | V2 |
| `PredictiveCodingNeuron` | Top-down prediction, bottom-up error | V2 |
| `SpikingNeuron` | Leaky integrate-and-fire, STDP-like update | V2 |

## V2 infrastructure

Modules that work with any network/neuron type:

- **Same-space propagation** — iterative refinement keeps all layers in the same vector space (no dimension change between layers).
- **Lateral inhibition** — top-k winners suppress others; sharpens competition between neurons.
- **Prediction-error depth signal** — stagnant layers on wrong predictions trigger depth growth.
- **Skip connections** — non-adjacent layer communication via `ConnectionTracker`. Auto-discovered and auto-pruned.
- **DA Selection** — proven MLE prototype selector: `cosine_sim(z, z_bar)`. Oracle gap O(1/sqrt(K)).
- **Sleep Consolidation** — hippocampal buffer + interleaved replay. Prevents catastrophic forgetting.
- **Theta Buffer** — sequence encoding via bigram co-occurrence. No positional embeddings.

## Benchmark results (sklearn digits, PCA-50)

Results from `compare.py` — all 5 neuron types start at `[50, 8, 8, 10]` and self-build.

### Growth dynamics

| Neuron type | Initial topology | Final topology (B1) | Width grows | Depth grows |
|-------------|-----------------|---------------------|-------------|-------------|
| Prototype   | [50, 8, 8, 10]  | [50, 341, 328, 10]  | ~660        | 0           |
| Perceptron  | [50, 8, 8, 10]  | [50, 341, 323, 10]  | ~650        | 0           |
| Dendritic   | [50, 8, 8, 10]  | [50, 345, 335, 10]  | ~670        | 0           |
| Predictive  | [50, 8, 8, 10]  | [50, 347, 342, 10]  | ~680        | 0           |
| Spiking     | [50, 8, 8, 10]  | [50, 512, 512, 10]  | ~1010       | 0           |

All neuron types self-size from 8 neurons per hidden layer to 300-512 neurons per layer, driven entirely by the error signal (split threshold = 0.50). Spiking neurons grow to the maximum (512) because their spike-based activations produce low similarity scores. Depth growth does not trigger on this dataset because adjacent-layer activations stay below the correlation threshold.

### Head-to-head

| Benchmark | Metric | Prototype | Perceptron | Dendritic | Predictive | Spiking |
|-----------|--------|-----------|------------|-----------|------------|---------|
| B1 Classification | Test accuracy | **92.5%** | 92.2% | 49.4% | 23.1% | 0.8% |
| B2 DA Theorem | K=5 gain | +1.4% | +1.4% | +1.4% | +1.4% | +1.4% |
| B3 Split MNIST | BWT (forgetting) | **-2.7%** | -3.1% | -4.3% | +3.8% | +0.0% |
| B4 Combined | Overall accuracy | **34%** | 33% | **35%** | 22% | 5% |
| B4 Combined | Sequence completion | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

### B1: Classification

Full training on all 10 digit classes, with sleep consolidation replay.

- **Prototype**: 92.5% accuracy, topology [50, 341, 328, 10], 0 dead neurons
- **Perceptron**: 92.2% accuracy, topology [50, 341, 323, 10], 0 dead neurons
- **Dendritic**: 49.4% accuracy, topology [50, 345, 335, 10], 0 dead neurons
- **Predictive**: 23.1% accuracy, topology [50, 347, 342, 10], 0 dead neurons
- **Spiking**: 0.8% accuracy, topology [50, 512, 512, 10], 0 dead neurons

Prototype and perceptron neurons lead on classification. Dendritic neurons show promise (multi-compartment structure captures patterns) but need tuning. Predictive coding and spiking neurons are functional but not yet competitive on raw accuracy -- their strengths are in different regimes (prediction error, temporal coding).

### B3: Continual learning (Split MNIST)

5 sequential tasks ([0,1], [2,3], [4,5], [6,7], [8,9]) with sleep between tasks.

- **Prototype**: avg 91.0%, BWT = -2.7% (mild forgetting, sleep consolidation helps)
- **Perceptron**: avg 89.9%, BWT = -3.1%
- **Dendritic**: avg 62.3%, BWT = -4.3%
- **Predictive**: BWT = +3.8% (positive transfer -- prediction-error signal prevents overwriting)
- **Spiking**: BWT = +0.0% (no forgetting, but near-chance accuracy)

### B4: Combined (one-shot + retention + sequence)

3 learning days with DA-selected prototypes and theta-buffer sequence completion.

- All 5 neuron types achieve **5/5** sequence completions via the theta buffer
- Dendritic leads on combined retention accuracy (35%), followed by prototype (34%) and perceptron (33%)

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
pytest tests/                                    # 80 tests
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
