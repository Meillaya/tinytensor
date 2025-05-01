#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifndef TENSOR_CU_H
#define TENSOR_CU_H

#define BLOCKSIZE 128
#define MAX_PREVS 3
#define MAX_ARGS 5
#define MAX_PARAM_TENSORS 10
// op codes (should be consistent with tensor.h)
#define MATMUL 0
#define MEAN 1
#define MUL 2
#define RELU 3
#define LOGSOFTMAX 4

// Forward declarations for structs to avoid circular dependency if tensor.h includes tensor_cu.h
// Alternatively, ensure one includes the other, or use a common types header.
typedef struct Arr Arr;
typedef struct Tensor Tensor;

// Struct definitions (can be identical to tensor.h or modified for CUDA specifics)
typedef struct Arr {
    float* values;      // CPU data (optional, might sync on demand)
    float* cuda_values; // GPU data
    int* shape;
    int* strides; // CPU copy, assumed constant after creation
    int ndim;
    int size;
} Arr;

typedef union {
    int ival;
    float fval;
    int* ilist;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad; // Gradient tensor, also lives on GPU
    int op;
    struct Tensor* prevs[MAX_PREVS]; // CPU pointers to previous tensors
    int num_prevs;
    Arg args[MAX_ARGS]; // CPU arguments
} Tensor;

// CUDA Utilities
cudaError_t checkCudaError(cudaError_t error, const char *file, int line);
// Macro for easier error checking
#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

// Memory Management
Arr* cuda_create_arr_zeros(int *shape, int ndim);
Arr* cuda_create_arr(float* host_data, int* shape, int ndim);
void cuda_free_arr(Arr* a);
Tensor* cuda_create_zero_tensor(int* shape, int ndim);
Tensor* cuda_create_tensor(float* host_data, int* shape, int ndim);
void cuda_free_tensor(Tensor* t);
void cuda_sync_tensor_data_to_host(Tensor* t); // GPU -> CPU
void cuda_sync_tensor_grad_to_host(Tensor* t); // GPU -> CPU
void cuda_ensure_data_on_gpu(Tensor* t);     // CPU -> GPU (if needed)
void cuda_ensure_grad_on_gpu(Tensor* t);     // CPU -> GPU (if needed)
void cuda_zero_grad(Tensor* t);             // Zero gradient on GPU

// CUDA Operations (Host functions launching kernels)
Tensor* cuda_matmul(Tensor* a, Tensor* b);
Tensor* cuda_relu(Tensor* inp);
Tensor* cuda_logsoftmax(Tensor* inp);
Tensor* cuda_mul(Tensor* a, Tensor* b);
Tensor* cuda_mean(Tensor* inp);

// CUDA Backward Operations (Host functions launching kernels)
void cuda_matmul_backward(Tensor* out);
void cuda_relu_backward(Tensor* out);
void cuda_logsoftmax_backward(Tensor* out);
void cuda_mul_backward(Tensor* out);
void cuda_mean_backward(Tensor* out);

// General Autograd
void cuda_backward(Tensor* t);

// Debugging
void cuda_print_tensor(Tensor* t, const char* name);

#endif // TENSOR_CU_H