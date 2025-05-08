# tinytensor

A small tensor library in C and CUDA, implementing basic tensor operations with autograd support.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/f0de68cd-dc7b-4592-b68a-265793c2c6f9">

## Description

This library provides fundamental tensor operations like creation, matrix multiplication, activation functions (ReLU, LogSoftmax), and automatic differentiation (backward pass). It includes separate implementations for CPU (in C) and GPU (using CUDA). An example demonstrates training a simple neural network on the MNIST dataset.

For a detailed explanation of the concepts and implementation, see the [tutorial](docs/tutorial.md).

## Building and Running

### Prerequisites
- C Compiler (like GCC)
- UV (Python package and project manager written in Rust)
- CUDA Toolkit (nvcc)
- Python (for MNIST data preparation)
- `make` (optional, if a Makefile is created)

### Build Steps

You can compile the examples manually:

```bash
# Compile the main CPU example
gcc -O3 -fopenmp -march=native -funroll-loops -Isrc -o src/main_cpu src/main.c src/tensor.c -lm

# Compile the CPU MNIST test
gcc -O3 -fopenmp -march=native -funroll-loops -Isrc -o tests/test_cpu tests/test.c src/tensor.c -lm

# Compile the CUDA MNIST test (requires tensor_cu.cu)
nvcc -O3 -Isrc -o tests/test_gpu tests/tests_cuda.cu src/tensor_cu.cu
```


### Running the MNIST Tests

1.  **Prepare MNIST Data:**
    ```bash
    uv run src/create_mnist_csv.py 
    ```
    This will generate `mnist_train.csv` and  `mnist_test.csv` in the root directory.

2.  **Run Tests:**
    ```bash
    # Run CPU test
    ./tests/test_cpu

    # Run CUDA test
    ./test_gpu 
    ```

### Running the Main Example
```bash
./main_cpu
```


# References
- [CUDA C++ Tensor Library](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
- [tensor.h](https://github.com/apoorvnandan/tensor.h) Tiny tensor library in C.