# Quick Start Guide: test_mla_helix_improved.py

## Overview

The improved MLA Helix test file provides a cleaner, more maintainable structure for testing MLA with Helix context parallelism.

## File Structure

```
tests/unittest/_torch/modules/
├── test_mla_helix_improved.py      # Main test file
├── mla_test_helpers.py             # Core helper classes
├── mla_weight_utils.py             # Weight generation utilities
└── test_utils/
    ├── __init__.py
    ├── error_reporting.py          # Tensor comparison utilities
    ├── rope_utils.py               # RoPE operations
    └── distributed_helpers.py      # MPI/distributed utilities
```

## Running Tests

### Basic Correctness Tests (Default)

```bash
# Run all basic correctness tests (5 scenarios)
pytest test_mla_helix_improved.py

# Run specific scenario
pytest test_mla_helix_improved.py::test_mla_helix_distributed[batch1_ctx4096]

# Run with verbose output
pytest -v test_mla_helix_improved.py
```

### Performance Benchmarks

```bash
# Run performance tests (marked with @pytest.mark.benchmark)
pytest -m benchmark test_mla_helix_improved.py

# Run with output capture disabled to see timing info
pytest -m benchmark -s test_mla_helix_improved.py
```

### Direct Execution (Manual Testing)

```bash
# Run with MPI directly (2 GPUs required)
mpirun -n 2 python test_mla_helix_improved.py

# This will run the first 3 scenarios with timing
```

## Understanding Test Scenarios

### Basic Scenarios (Always Run)

```python
BASIC_SCENARIOS = [
    TestScenario(batch=1, ctx_len=64),       # Tiny - quick sanity check
    TestScenario(batch=1, ctx_len=4096),     # Small - standard test
    TestScenario(batch=1, ctx_len=32768),    # Medium - longer context
    TestScenario(batch=8, ctx_len=4096),     # Multi-batch small
    TestScenario(batch=8, ctx_len=16384),    # Multi-batch medium
]
```

### Extended Scenarios (Optional)

14 additional scenarios covering more combinations. Add to test by modifying parametrize decorator.

## Customizing Tests

### Add a Custom Scenario

```python
# In test_mla_helix_improved.py, add to BASIC_SCENARIOS:

CUSTOM_SCENARIO = TestScenario(
    batch=4,           # Batch size
    ctx_len=8192,      # Context length
    atol=1e-1,         # Absolute tolerance
    rtol=5e-2,         # Relative tolerance
    # ... other parameters use defaults
)
```

### Adjust Tolerance

```python
# Per test
@pytest.mark.parametrize("scenario", BASIC_SCENARIOS)
def test_mla_helix_distributed(scenario: TestScenario, max_mismatch_ratio: float = 0.05):
    # Now allows 5% mismatches instead of 2%
    ...

# Or modify scenario directly
TestScenario(batch=1, ctx_len=64, atol=1e-2, rtol=1e-3)
```

### Run Specific Batch Sizes

```bash
# Only batch=1 scenarios
pytest test_mla_helix_improved.py -k "batch1"

# Only large context scenarios
pytest test_mla_helix_improved.py -k "ctx32768"
```

## Understanding Test Components

### TestScenario

Configuration for a single test case. Controls:
- Model architecture (num_heads, hidden_size, etc.)
- RoPE parameters
- Test dimensions (batch, ctx_len)
- Tolerances (atol, rtol)

### KVCacheSetup

Manages KV cache initialization for both reference and distributed runs.

```python
kv_cache_setup = KVCacheSetup(scenario, mapping, gen_steps)
# ... use kv_cache_setup.kv_cache_manager
# ... use kv_cache_setup.attn_metadata
kv_cache_setup.shutdown()  # Cleanup
```

### MLADistributedRunner

Orchestrates distributed MLA testing:

```python
runner = MLADistributedRunner(scenario, rank, world_size, gen_steps)
mla = runner.create_mla_model(pos_embd_params)
runner.load_distributed_weights(mla, weights)
outputs = runner.run_generation_steps(...)
ratio = runner.compare_with_reference(outputs, ref_output)
```

### ReferenceMLARunner

Runs single-GPU reference:

```python
ref_runner = ReferenceMLARunner(scenario, gen_steps)
ref_output = ref_runner.run(mla, input_ctx, input_gen, position_ids_ctx)
```

## Debugging

### Enable Verbose Output

```python
# Add print statements in MLADistributedRunner methods
def run_generation_steps(self, ...):
    print(f"Rank {self.rank}: Starting generation...")
    for step in range(self.scenario.ref_steps):
        print(f"Rank {self.rank}: Step {step}")
        ...
```

### Check Tensor Values

```python
# In test_mla_helix_improved.py, after outputs are generated:
print(f"Output shape: {outputs[0].shape}")
print(f"Output stats: mean={outputs[0].mean()}, std={outputs[0].std()}")
print(f"First values: {outputs[0][0, :10]}")
```

### Test Single Component

```python
# Test KV cache setup independently
from mla_test_helpers import KVCacheSetup, TestScenario

scenario = TestScenario(batch=1, ctx_len=64)
mapping = Mapping(world_size=1, rank=0)
kv_setup = KVCacheSetup(scenario, mapping, gen_steps=1)
print(f"KV cache created: {kv_setup.kv_cache_manager}")
```

## Common Issues

### "needs 2 GPUs to run this test"

- Test requires 2 GPUs minimum
- Check: `nvidia-smi` or `torch.cuda.device_count()`
- Solution: Run on multi-GPU system or skip with `-k "not helix"`

### MPI Import Errors

- Requires `mpi4py` installed
- Solution: `pip install mpi4py`

### Mismatch Ratio Too High

- Some scenarios naturally have higher numerical error (especially bf16)
- Solution: Adjust `max_mismatch_ratio` parameter or tolerances in TestScenario

### CUDA Out of Memory

- Large scenarios (ctx_len > 100k) may OOM on smaller GPUs
- Solution: Use smaller scenarios or larger GPUs

## Advanced Usage

### Adding Performance Metrics

```python
# In MLADistributedRunner.run_generation_steps:
import time

start = time.time()
for step in range(self.scenario.ref_steps):
    # ... existing code ...
    pass
end = time.time()

throughput = self.scenario.batch * self.scenario.ref_steps / (end - start)
print(f"Throughput: {throughput:.2f} tokens/sec")
```

### Testing Different World Sizes

```python
# Modify test to use world_size=4
@pytest.mark.parametrize("world_size", [2, 4])
def test_mla_helix_distributed(scenario, world_size):
    with MPIPoolExecutor(max_workers=world_size) as executor:
        ...
```

### Integration with CI/CD

```yaml
# .github/workflows/test.yml
- name: Run MLA Helix Tests
  run: |
    pytest tests/unittest/_torch/modules/test_mla_helix_improved.py \
      -v --tb=short --maxfail=3
```

## Tips

1. **Start small**: Test with `batch=1, ctx_len=64` first
2. **Use verbose mode**: `pytest -v -s` to see all output
3. **Test incrementally**: Add one scenario at a time
4. **Check tolerances**: Adjust `atol`/`rtol` based on dtype and model
5. **Profile performance**: Use `pytest -m benchmark` for timing tests

## Getting Help

- Check `REFACTORING_SUMMARY.md` for design overview
- Check `COMPARISON.md` for differences from original
- Read docstrings in helper classes for detailed explanations
- Examine original `test_mla_helix.py` for reference implementation


