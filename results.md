# Experiment 6 – Results

## Sample Results

| Operation | Matrix Size | Execution Time | Performance |
|---|---:|---:|---:|
| Uncoalesced Transpose | 4096 | 1.850 ms | 145.1 GB/s |
| Coalesced Transpose | 4096 | 0.720 ms | 372.8 GB/s |
| Naive Matrix Multiplication | 1024 | 3.900 ms | 550.6 GFLOPS |

## Speedup

Speedup = Uncoalesced Time / Coalesced Time

Speedup = 1.850 / 0.720 = 2.57×

## Verification
- Uncoalesced transpose: PASS
- Coalesced transpose: PASS
- Naive matrix multiplication: PASS

## Observation
The coalesced transpose performs faster than the uncoalesced transpose because it uses shared memory for efficient memory access.

Note: These are illustrative values from the lab manual. Replace them with actual results from your GPU execution.
