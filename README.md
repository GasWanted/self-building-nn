# self-building-nn

Neural networks that build their own architecture at runtime.

No hand-designed topology. No evolutionary generations. The network starts with 2 small hidden layers and grows width (new neurons) and depth (new layers) driven by two biological signals:

- **Error signal**: layer lacks coverage for an input pattern -> add a neuron (width)
- **Information signal**: adjacent layers are too correlated -> duplicate a layer (depth)
- **Prune**: neurons that stop firing get removed. Empty layers get removed.

Width and depth growth are the same mechanism at different scales: **mitosis**. Duplicate a neuron, duplicate a layer. The copies diverge through learning.

## Neuron types

The growth engine is neuron-agnostic. Swap the neuron type and rerun benchmarks.

| Type | Description | Status |
|------|-------------|--------|
| `PrototypeNeuron` | Competitive learning, cosine similarity | V1 |
| `PerceptronNeuron` | Standard w*x+b, gradient update | V1 (baseline) |
| Spiking (LIF) | Spike timing, STDP | Planned |
| Dendritic | Multi-compartment, per-dendrite nonlinearity | Planned |
| Predictive coding | Top-down prediction, bottom-up error | Planned |

## Infrastructure

Modules that work with any network/neuron type:

- **DA Selection** — proven MLE prototype selector: `cosine_sim(z, z_bar)`. Oracle gap O(1/sqrt(K)).
- **Sleep Consolidation** — hippocampal buffer + interleaved replay. Prevents catastrophic forgetting.
- **Theta Buffer** — sequence encoding via bigram co-occurrence. No positional embeddings.

## Benchmark results (sklearn digits, PCA-50)

Results from `compare.py` — both neuron types start at `[50, 8, 8, 10]` and self-build.

### Growth dynamics

| Neuron type | Initial topology | Final topology (B1) | Width grows | Depth grows |
|-------------|-----------------|---------------------|-------------|-------------|
| Prototype   | [50, 8, 8, 10]  | [50, 337, 212, 10]  | ~540        | 0           |
| Perceptron  | [50, 8, 8, 10]  | [50, 338, 222, 10]  | ~550        | 0           |

The network self-sizes from 8 neurons per hidden layer to 200-340 neurons per layer, driven entirely by the error signal (split threshold = 0.50). Depth growth does not trigger on this dataset because adjacent-layer activations stay below the correlation threshold.

### Head-to-head

| Benchmark | Metric | Prototype | Perceptron | Winner |
|-----------|--------|-----------|------------|--------|
| B1 Classification | Test accuracy | **87.2%** | 84.4% | Proto |
| B2 DA Theorem | K=5 gain | +1.4% | +1.4% | Tie |
| B3 Split MNIST | BWT (forgetting) | **-7.7%** | -7.9% | Proto |
| B3 Split MNIST | Final avg accuracy | 88.6% | **89.5%** | Perc |
| B4 Combined | Overall accuracy | **28.4%** | 22.4% | Proto |
| B4 Combined | Sequence completion | 5/5 | 5/5 | Tie |

### B1: Classification

Full training on all 10 digit classes, with sleep consolidation replay.

- **Prototype**: 87.2% accuracy, topology [50, 337, 212, 10], 0 dead neurons
- **Perceptron**: 84.4% accuracy, topology [50, 338, 222, 10], 0 dead neurons

### B3: Continual learning (Split MNIST)

5 sequential tasks ([0,1], [2,3], [4,5], [6,7], [8,9]) with sleep between tasks.

- **Prototype**: avg 88.6%, BWT = -7.7% (mild forgetting, sleep consolidation helps)
- **Perceptron**: avg 89.5%, BWT = -7.9%

### B4: Combined (one-shot + retention + sequence)

3 learning days with DA-selected prototypes and theta-buffer sequence completion.

- Both achieve **5/5** sequence completions via the theta buffer
- Prototype leads on retention accuracy (28% vs 22%)

### B5: Ablation

Each infrastructure component tested in isolation and combination:

| Config | Prototype | Perceptron |
|--------|-----------|------------|
| Baseline (nothing) | 12% | 13% |
| + DA selection | 17% | 17% |
| + Sleep only | 17% | 20% |
| + Theta only | 14% | 12% |
| DA + Sleep | 30% | 26% |
| Full (DA+Sleep+Theta) | 20% | 17% |

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
pytest tests/                          # 40 tests
PYTHONPATH=. python3 run.py            # single neuron-type benchmark suite
PYTHONPATH=. python3 compare.py        # prototype vs perceptron head-to-head
```

## Structure

```
src/
  neurons/          # pluggable neuron types
  network/          # layer + self-building network (growth engine)
  signals/          # error (width) + information (depth) triggers
  infrastructure/   # DA selection, sleep, theta buffer
  benchmarks/       # B1-B5 + harness
  data/             # MNIST loader
tests/
docs/superpowers/specs/
```
