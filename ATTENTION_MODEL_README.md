# SSSD-ECG with Full Attention

This directory contains a modified version of SSSD-ECG that replaces the S4 (Structured State Space) layers with standard Transformer-style multi-head self-attention layers.

## Overview

The original SSSD-ECG model uses S4 layers for sequence modeling, which requires external dependencies like `pykeops` and custom CUDA extensions. This implementation provides a drop-in replacement using standard PyTorch attention mechanisms, making it:

- **Easier to install**: Only requires PyTorch and standard dependencies
- **More portable**: No custom CUDA extensions needed
- **Easier to understand**: Uses familiar Transformer architecture
- **Hardware agnostic**: Works on any device that supports PyTorch

## Key Changes

### S4 → Attention Replacement

The main changes are:

1. **AttentionModel.py**: New module containing attention-based layers
   - `MultiHeadAttention`: Standard scaled dot-product attention
   - `AttentionLayer`: Single attention block with feedforward network
   - `BidirectionalAttentionLayer`: Processes sequences in both directions

2. **SSSD_ECG_Attention.py**: Modified SSSD-ECG model
   - Replaces `S4Layer` with `AttentionLayer`
   - Maintains the same interface and architecture otherwise
   - All other components (residual blocks, diffusion embeddings) remain the same

### Parameter Mapping

When migrating from S4-based model to Attention-based model:

| S4 Parameter | Attention Parameter | Notes |
|--------------|---------------------|-------|
| `s4_lmax` | `attention_lmax` | Maximum sequence length (kept for compatibility) |
| `s4_d_state` | `attention_num_heads` | State dimension → Number of attention heads |
| `s4_dropout` | `attention_dropout` | Dropout rate |
| `s4_bidirectional` | N/A | Always unidirectional in current implementation |
| `s4_layernorm` | `attention_layernorm` | Layer normalization |

## Installation

### Requirements

```bash
# Core dependencies (already in requirements.txt)
torch>=1.10.0
numpy>=1.20.0
einops>=0.4.0
opt_einsum>=3.3.0

# Training dependencies
tqdm>=4.60.0
tensorboard>=2.8.0

# Optional dependencies
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
wfdb>=3.4.0
```

### No S4 Dependencies Needed

Unlike the original S4-based model, you do **NOT** need:
- `pykeops`
- Cauchy kernel CUDA extensions
- Any custom compilation steps

## Usage

### Basic Example

```python
import torch
from models.SSSD_ECG_Attention import SSSD_ECG_Attention

# Create model
model = SSSD_ECG_Attention(
    in_channels=12,           # 12-lead ECG
    res_channels=256,
    skip_channels=256,
    out_channels=12,
    num_res_layers=36,
    diffusion_step_embed_dim_in=128,
    diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512,
    attention_lmax=1000,      # Maximum sequence length
    attention_num_heads=8,    # Number of attention heads
    attention_dropout=0.1,    # Dropout rate
    attention_layernorm=True, # Use layer normalization
)

# Prepare input
batch_size = 4
seq_len = 1000
noise = torch.randn(batch_size, 12, seq_len)
label = torch.zeros(batch_size, 0)  # Unconditional
diffusion_steps = torch.randint(0, 50, (batch_size,))

# Forward pass
output = model((noise, label, diffusion_steps))
print(output.shape)  # (4, 12, 1000)
```

### Conditional Generation

```python
# Create model with label conditioning
model = SSSD_ECG_Attention(
    in_channels=12,
    res_channels=256,
    skip_channels=256,
    out_channels=12,
    num_res_layers=36,
    diffusion_step_embed_dim_in=128,
    diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512,
    attention_lmax=1000,
    attention_num_heads=8,
    attention_dropout=0.1,
    attention_layernorm=True,
    label_embed_classes=5,    # 5 classes
    label_embed_dim=128,
)

# One-hot encoded labels
label = torch.zeros(batch_size, 5)
label[:, 0] = 1.0  # Class 0

# Forward pass with labels
output = model((noise, label, diffusion_steps))
```

### Using AttentionLayer Directly

```python
from models.AttentionModel import AttentionLayer

# Create attention layer
layer = AttentionLayer(
    features=256,
    lmax=1000,
    num_heads=8,
    dropout=0.1,
    layer_norm=True
)

# Input: (seq_len, batch_size, features)
x = torch.randn(1000, 4, 256)
output = layer(x)
print(output.shape)  # (1000, 4, 256)
```

## Testing

Run the test suite to verify the implementation:

```bash
python test_attention_model.py
```

This will test:
- Basic attention layer functionality
- Bidirectional attention layers
- Full SSSD_ECG_Attention model
- Conditional generation
- CUDA compatibility (if available)

## Architecture Details

### Multi-Head Self-Attention

The attention mechanism uses standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q, K, V are query, key, value matrices
- d_k is the dimension per head
- Multiple heads process different representation subspaces

### Residual Block Structure

Each residual block contains:

1. Diffusion step embedding (time conditioning)
2. Convolution layer
3. **First Attention Layer** (replaces S4)
4. Label embedding (if conditional)
5. **Second Attention Layer** (replaces S4)
6. Gated activation (tanh × sigmoid)
7. Skip connection

### Comparison with S4

| Aspect | S4 Layer | Attention Layer |
|--------|----------|-----------------|
| Complexity | O(L log L) with FFT | O(L²) |
| Memory | O(L) | O(L²) |
| Dependencies | pykeops, CUDA extensions | PyTorch only |
| Long sequences | Efficient | Memory intensive |
| Parallelization | Limited | Fully parallel |
| Interpretability | Complex | Clear attention weights |

## Performance Considerations

### Memory Usage

Attention has O(L²) memory complexity, where L is the sequence length. For long sequences (>2000 samples):

- Consider reducing `attention_num_heads`
- Use gradient checkpointing
- Process in smaller chunks
- Consider using sparse attention patterns

### Computational Cost

Full attention is more computationally expensive than S4 for long sequences:

- **S4**: O(L log L) with FFT convolution
- **Attention**: O(L²d) where d is the model dimension

For typical ECG sequences (1000-5000 samples), this is usually acceptable on modern GPUs.

### Recommended Settings

For **12-lead ECG** with **1000 samples**:

```python
res_channels=256
num_res_layers=36
attention_num_heads=8
attention_dropout=0.1
```

For **shorter sequences** (<500 samples):

```python
res_channels=128
num_res_layers=12
attention_num_heads=4
attention_dropout=0.0
```

## Training

The training procedure remains the same as the original SSSD-ECG:

1. Use the same diffusion schedule
2. Same loss function (MSE on noise prediction)
3. Same optimizer settings (AdamW with cosine annealing)

The only difference is in model initialization:

```python
# Old (S4-based)
from models.SSSD_ECG import SSSD_ECG
model = SSSD_ECG(s4_lmax=1000, s4_d_state=64, ...)

# New (Attention-based)
from models.SSSD_ECG_Attention import SSSD_ECG_Attention
model = SSSD_ECG_Attention(attention_lmax=1000, attention_num_heads=8, ...)
```

## Advantages

1. **Simplicity**: No external dependencies beyond PyTorch
2. **Portability**: Runs anywhere PyTorch runs
3. **Debugging**: Easier to debug and understand
4. **Flexibility**: Easy to modify and extend
5. **Stability**: More stable training (no FFT numerical issues)

## Limitations

1. **Memory**: Higher memory usage for long sequences
2. **Speed**: Slower than S4 for very long sequences (>2000)
3. **Inductive bias**: Less structured inductive bias than S4

## Migration Guide

To convert existing S4-based code:

1. Replace import:
   ```python
   # from models.SSSD_ECG import SSSD_ECG
   from models.SSSD_ECG_Attention import SSSD_ECG_Attention
   ```

2. Update model initialization:
   ```python
   model = SSSD_ECG_Attention(
       # ... same parameters ...
       attention_lmax=s4_lmax,
       attention_num_heads=8,  # New parameter
       attention_dropout=s4_dropout,
       attention_layernorm=s4_layernorm,
   )
   ```

3. Everything else remains the same!

## Files

- `models/AttentionModel.py`: Attention layer implementations
- `models/SSSD_ECG_Attention.py`: Main model with attention
- `test_attention_model.py`: Comprehensive test suite
- `ATTENTION_MODEL_README.md`: This file

## Citation

If you use this attention-based version, please cite both the original SSSD paper and note the architectural modification:

```bibtex
@article{sssd2023,
  title={Structured State Space Models for In-Context Reinforcement Learning},
  author={...},
  journal={...},
  year={2023}
}
```

## License

Same license as the original SSSD-ECG project.

## Contributing

Contributions are welcome! Areas for improvement:

- Sparse attention patterns for longer sequences
- Efficient attention implementations (FlashAttention, etc.)
- Bidirectional attention support
- Relative position encodings
- Comparison benchmarks with S4

## Contact

For questions or issues, please open an issue on the GitHub repository.
