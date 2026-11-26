# Experiment 6: Memory Optimization in CUDA

## 1. Aim

To study how global memory access patterns affect GPU performance by comparing coalesced and uncoalesced matrix transpose, performing naive matrix multiplication, and measuring execution time.

## 2. Introduction

CUDA is a parallel computing platform developed by NVIDIA that allows programmers to use GPUs for high-performance computations.

GPU performance depends not only on the number of computations but also on how efficiently data is accessed from memory.

In this experiment, different memory access patterns are studied to understand their effect on GPU performance.

## 3. Objectives

- Understand memory coalescing in CUDA.
- Compare coalesced and uncoalesced memory access.
- Implement matrix transpose using CUDA kernels.
- Understand the use of shared memory.
- Perform naive matrix multiplication.
- Measure execution time and GPU performance.

## 4. Theory

### 4.1 Memory Coalescing

Memory coalescing is a technique in which memory requests from neighbouring GPU threads are combined into fewer memory transactions.

**Coalesced Access:**
- Threads access consecutive memory locations.
- Requires fewer memory transactions.
- Improves memory bandwidth utilization.

**Uncoalesced Access:**
- Threads access scattered memory locations.
- Requires more memory transactions.
- Reduces memory bandwidth efficiency.

### 4.2 Matrix Transpose

Matrix transpose is the process of converting rows into columns and columns into rows.

For an input matrix:

A[i][j]

The transposed output is:

B[j][i]

Two approaches are implemented:

**Uncoalesced Transpose:**

Each CUDA thread copies one matrix element. Although memory reads are consecutive, memory writes are separated by a stride, resulting in uncoalesced access.

**Coalesced Transpose:**

Shared memory is used to temporarily store a 32 × 32 matrix tile. After synchronization, the data is written in transposed order to achieve coalesced global-memory access.

### 4.3 Shared Memory

Shared memory is a fast memory accessible by threads within the same CUDA block.

It is used to temporarily store matrix elements and improve memory-access efficiency.

The shared-memory tile is declared as:

`__shared__ float tile[32][33];`

The additional column helps avoid shared-memory bank conflicts.

### 4.4 Naive Matrix Multiplication

Matrix multiplication calculates the product of two matrices.

For matrices A and B:

C = A × B

Each CUDA thread calculates one element of the output matrix using repeated global-memory accesses.

This implementation does not use shared-memory tiling for data reuse.

## 5. Tools and Technologies

- Programming Language: CUDA C/C++
- GPU: NVIDIA GPU
- Compiler: NVCC
- Platform: CUDA Toolkit / Google Colab GPU
- Memory: Global Memory and Shared Memory

## 6. Procedure

1. Check GPU availability using `nvidia-smi`.
2. Verify CUDA compiler installation using `nvcc --version`.
3. Create the CUDA source file `exp6.cu`.
4. Implement the uncoalesced matrix transpose kernel.
5. Implement the coalesced matrix transpose kernel using shared memory.
6. Implement the naive matrix multiplication kernel.
7. Allocate host and device memory.
8. Copy input data from CPU memory to GPU memory.
9. Execute the CUDA kernels.
10. Measure execution time using CUDA events.
11. Copy the results back to CPU memory.
12. Verify GPU results against CPU reference calculations.
13. Compare execution time and performance.

## 7. Implementation Details

### Part A: Matrix Transpose

- Input matrix size: 4096 × 4096
- Block size: 32 × 32
- Two transpose kernels are executed.
- Each kernel is repeated 20 times for timing.
- Average execution time is calculated.

### Part B: Matrix Multiplication

- Input matrix size: 1024 × 1024
- Block size: 16 × 16
- Each thread calculates one output element.
- Kernel execution is repeated five times for timing.
- Results are verified against CPU reference calculations.

## 8. Performance Metrics

**Execution Time:** Measures the time taken by the GPU to execute a kernel.

**Memory Bandwidth:** Measures the amount of data transferred per second, expressed in GB/s.

**GFLOPS:** Measures billions of floating-point operations performed per second.

**Speedup:** Measures how much faster the coalesced transpose is compared to the uncoalesced transpose.

Speedup = Uncoalesced Execution Time / Coalesced Execution Time

## 9. Results and Observations

The following are illustrative sample results from the lab manual.

| Operation | Execution Time | Performance |
|---|---:|---:|
| Uncoalesced Transpose | 1.850 ms | 145.1 GB/s |
| Coalesced Transpose | 0.720 ms | 372.8 GB/s |
| Naive Matrix Multiplication | 3.900 ms | 550.6 GFLOPS |

### Speedup Calculation

Speedup = 1.850 / 0.720

Speedup = 2.57×

### Observations

1. Coalesced transpose performs faster than uncoalesced transpose.
2. Shared memory improves global-memory access efficiency.
3. Uncoalesced access requires more memory transactions.
4. Naive matrix multiplication performs repeated global-memory accesses.
5. Both transpose implementations and matrix multiplication pass the correctness checks in the sample output.

## 10. Conclusion

The experiment demonstrates that memory access patterns significantly affect GPU performance.

Coalesced memory access improves performance by reducing unnecessary global-memory transactions. Shared memory helps organize data efficiently during matrix transpose.

The experiment also demonstrates that naive matrix multiplication can be improved further using shared-memory tiling and data reuse.

Thus, efficient memory access is an important factor in CUDA GPU optimization.

## 11. References

- Experiment 6: Memory Optimization in CUDA – Lab Manual.
- NVIDIA CUDA C++ Programming Guide.
- NVIDIA CUDA Toolkit Documentation.
