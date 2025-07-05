# DSMNet Multi-GPU Implementation

This branch contains minimal modifications to enable multi-GPU training for DSMNet using TensorFlow's `MirroredStrategy`.

## Quick Setup

### 1. Configure Multi-GPU Settings

Edit `config.py` to configure your multi-GPU setup:

```python
# For Multi-GPU training
multi_gpu_enabled = True
gpu_devices = "0,1"  # Use GPUs 0 and 1

# For Single-GPU training (original behavior)
multi_gpu_enabled = False
gpu_devices = "0"    # Use only GPU 0
```

### 2. Test Configuration

Run the test script to verify your setup:

```bash
python test_multi_gpu.py
```

### 3. Run Training

Use the original training script:

```bash
./run_train_test.sh
```

## What Changed

### Minimal Modifications Made:

1. **config.py**: Added multi-GPU configuration variables and TensorFlow strategy setup
2. **train_mtl.py**: Wrapped model creation in strategy scope and added distributed training step
3. **train_dae.py**: Same multi-GPU modifications for DAE training
4. **run_train_test.sh**: Added usage instructions
5. **test_multi_gpu.py**: New test script to verify configuration

### Architecture Preservation:

- ✅ Original model architecture unchanged
- ✅ Original training logic preserved
- ✅ Backward compatibility with single-GPU training
- ✅ No redundant files or complex abstractions
- ✅ Minimal code changes

## How It Works

- **Single GPU**: When `multi_gpu_enabled = False`, the code runs exactly as before
- **Multi-GPU**: When `multi_gpu_enabled = True`, TensorFlow's `MirroredStrategy` automatically:
  - Replicates the model across available GPUs
  - Distributes batches across GPUs
  - Synchronizes gradients across replicas
  - Scales the effective batch size by number of GPUs

## Performance Notes

- Effective batch size becomes `batch_size * num_gpus`
- Memory usage is distributed across GPUs
- Training speed should scale near-linearly with number of GPUs
- Best results with batch sizes that divide evenly across GPUs

## Troubleshooting

1. **GPU Memory Issues**: Reduce batch size in config.py
2. **CUDA Out of Memory**: Check GPU availability with `nvidia-smi`
3. **Single Replica Warning**: Verify GPU devices are visible and available
4. **Performance Issues**: Ensure adequate CPU cores and memory bandwidth

## Reverting to Single-GPU

Simply set `multi_gpu_enabled = False` in `config.py` - no other changes needed.
