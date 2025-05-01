# tinytensor

A small tensor library in C and CUDA, implementing basic tensor operations with autograd support.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/f0de68cd-dc7b-4592-b68a-265793c2c6f9">

## Description

This library provides fundamental tensor operations like creation, matrix multiplication, activation functions (ReLU, LogSoftmax), and automatic differentiation (backward pass). It includes separate implementations for CPU (in C) and GPU (using CUDA). An example demonstrates training a simple neural network on the MNIST dataset.

For a detailed explanation of the concepts and implementation, see the [tutorial](docs/tutorial.md).

## Building and Running

### Prerequisites
- C Compiler (like GCC)
- CUDA Toolkit (nvcc)
- Python (for MNIST data preparation)
- `make` (optional, if a Makefile is created)

### Build Steps

You can compile the examples manually:

```bash
# Compile the main CPU example
gcc src/main.c src/tensor.c -o main_cpu -lm 

# Compile the CPU MNIST test
gcc tests/test.c src/tensor.c -o test_cpu -lm

# Compile the CUDA MNIST test (requires tensor_cu.cu)
nvcc tests/tests_cuda.cu src/tensor_cu.cu -o test_gpu -lcudart 
```
*(Replace `src/tensor.c` and `src/tensor_cu.cu` if your implementation files have different names)*

### Running the MNIST Tests

1.  **Prepare MNIST Data:**
    ```bash
    python src/create_mnist_csv.py 
    ```
    This will generate `mnist_train.csv` (and potentially `mnist_test.csv`) in the root directory.

2.  **Run Tests:**
    ```bash
    # Run CPU test
    ./test_cpu

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