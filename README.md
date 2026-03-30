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

## Benchmarks (B1-B5)

| # | Test | Measures |
|---|------|----------|
| B1 | Full MNIST classification | Accuracy, topology, dead neurons |
| B2 | DA theorem verification | Prototype selection gain vs K |
| B3 | Split MNIST (5 tasks) | Backward transfer (forgetting) |
| B4 | Combined task | One-shot + retention + sequence |
| B5 | Ablation | Marginal contribution of each component |

## Quick start

```bash
pip install -r requirements.txt
pytest tests/
python -m src.benchmarks.harness  # run full suite
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
