# transformer_robomimic

A drop-in replacement for the standard [robomimic](https://robomimic.github.io/) library, designed for use inside the classical **IsaacLab conda environment**. This fork adds native support for ACT-style transformer policies with configurable context history and action chunking — all controlled directly through the existing robomimic JSON config format.

---

## Why this exists

Vanilla robomimic does not support transformer-based policy architectures with action chunking out of the box. This fork extends it to allow you to train and evaluate ACT (Action Chunking with Transformers) style models without leaving the IsaacLab ecosystem or changing your existing training pipeline.

---

## Features

- Drop-in replacement for standard robomimic in IsaacLab conda environments
- Transformer policy head configurable entirely via JSON
- ACT-style **action chunking**: predict a chunk of future actions in one forward pass
- Configurable **context history length** for observation sequences
- Standard robomimic workflows and interfaces remain unchanged

---

## Installation

Replace your existing robomimic installation inside your IsaacLab conda environment:

```bash
conda activate isaaclab  # or your environment name
pip uninstall robomimic
git clone https://github.com/mschmotzer/transformer_robomimic.git
cd transformer_robomimic
pip install -e .
```

---

## Configuration

The transformer policy is configured through the standard robomimic JSON config file. Add or modify the `"transformer"` block inside your model config:

```json
{
  "transformer": {
    "enabled": true,
    "supervise_all_steps": true,
    "num_layers": 8,
    "context_length": 2,
    "embed_dim": 512,
    "num_heads": 8,
    "activation": "gelu",
    "action_chunk_size": 40
  }
}
```

### Parameter reference

| Parameter | Type | Description |
|---|---|---|
| `enabled` | `bool` | Set to `true` to activate the transformer policy head. |
| `supervise_all_steps` | `bool` | If `true`, compute loss over all predicted steps in the chunk, not just the first. |
| `num_layers` | `int` | Number of transformer encoder/decoder layers. |
| `context_length` | `int` | Number of past observation timesteps fed as context to the transformer. |
| `embed_dim` | `int` | Token embedding dimensionality. |
| `num_heads` | `int` | Number of attention heads. Must evenly divide `embed_dim`. |
| `activation` | `str` | Activation function used in the feed-forward layers (`"gelu"` or `"relu"`). |
| `action_chunk_size` | `int` | Number of future action steps predicted in a single forward pass (action chunk). |

---

## How it works

### Action chunking

Instead of predicting a single action per timestep, the model outputs a sequence of `action_chunk_size` future actions in one forward pass. At inference time, you can either execute the full chunk open-loop or re-plan at every step.

### Context history

`context_length` controls how many past observation frames are passed as input tokens to the transformer. A value of `1` means only the current observation is used; higher values allow the model to reason over recent history.

### Supervision

When `supervise_all_steps` is `true`, the training loss is computed over every step in the predicted chunk against the corresponding ground-truth actions. When `false`, only the first predicted action is supervised.

---

## Example config

Below is a minimal working config snippet for an ACT-style policy with a 40-step action chunk and 2-step context window:

```json
{
  "algo_name": "bc",
  "algo": {
    "transformer": {
      "enabled": true,
      "supervise_all_steps": true,
      "num_layers": 8,
      "context_length": 2,
      "embed_dim": 512,
      "num_heads": 8,
      "activation": "gelu",
      "action_chunk_size": 40
    }
  }
}
```

Pass your config as usual:

```bash
python train.py --config /path/to/your_config.json
```

---

## Compatibility

- Designed for the **IsaacLab classical conda environment**
- Requires Python ≥ 3.8
- PyTorch ≥ 1.12 recommended
- Fully compatible with existing robomimic datasets and data loaders

---

## References

- [robomimic](https://robomimic.github.io/) — the upstream library this project extends
- [ACT: Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/) — the method this fork is inspired by
- [IsaacLab](https://isaac-sim.github.io/IsaacLab/) — the simulation framework this targets
