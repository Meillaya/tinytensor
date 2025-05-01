#include "tensor_cu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Utilities Implementation
cudaError_t checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        // Consider not exiting immediately in a library, maybe return the error.
        exit(EXIT_FAILURE); 
    }
    // cudaDeviceSynchronize(); // Optional: Synchronize for more precise error location, but slows down.
    return error;
}

// CUDA kernel definitions
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // Assume row-major layout for A, B, C
        // A: M x K, B: K x N, C: M x N
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// dA = dC @ B.T
__global__ void matmul_transpose_B_kernel(float* dC, float* B, float* dA, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to M (dA rows)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to K (dA cols)
    
    if (row < M && col < K) {
        float sum = 0.0f;
         // dC: M x N, B: K x N, dA: M x K
        for (int n = 0; n < N; n++) {
            // B[n, col] in B transposed is B[col, n] in original B (K x N)
            sum += dC[row * N + n] * B[col * N + n]; 
        }
        atomicAdd(&dA[row * K + col], sum); // Accumulate gradients
    }
}

// dB = A.T @ dC
__global__ void transpose_A_matmul_kernel(float* A, float* dC, float* dB, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Corresponds to K (dB rows)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to N (dB cols)
    
    if (row < K && col < N) {
        float sum = 0.0f;
        // A: M x K, dC: M x N, dB: K x N
        for (int m = 0; m < M; m++) {
            // A[m, row] in A transposed is A[row, m] in original A (M x K)
            sum += A[m * K + row] * dC[m * N + col];
        }
        atomicAdd(&dB[row * N + col], sum); // Accumulate gradients
    }
}

__global__ void relu_kernel(float* out, const float* inp, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(inp[i], 0.0f);
    }
}

__global__ void relu_backward_kernel(float* dinp, const float* dout, const float* inp_val, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&dinp[i], (inp_val[i] > 0.0f) ? dout[i] : 0.0f);
    }
}

__global__ void logsoftmax_kernel(float* out, const float* inp, int B, int C, int stride_B, int stride_C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        const float* current_inp_row = inp + b * stride_B;
        float* current_out_row = out + b * stride_B; // Assuming output has same strides for simplicity

        float max_val = -INFINITY;
        // Find max value in the row
        for (int c = 0; c < C; c++) {
            max_val = fmaxf(max_val, current_inp_row[c * stride_C]);
        }
        
        // Compute sum of exp(val - max_val)
        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_exp += expf(current_inp_row[c * stride_C] - max_val);
        }
        float log_sum_exp = logf(sum_exp);
        
        // Compute log softmax
        for (int c = 0; c < C; c++) {
            current_out_row[c * stride_C] = current_inp_row[c * stride_C] - max_val - log_sum_exp;
        }
    }
}

__global__ void logsoftmax_backward_kernel(float* dinp, const float* dout, const float* out_val, int B, int C, int stride_B, int stride_C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float* current_dinp_row = dinp + b * stride_B;
        const float* current_dout_row = dout + b * stride_B;
        const float* current_out_val_row = out_val + b * stride_B;

        float sum_dout = 0.0f;
        // First compute sum of incoming gradients for this row
        for (int c = 0; c < C; c++) {
            sum_dout += current_dout_row[c * stride_C];
        }

        // Compute gradient for input
        // dL/dx_i = dL/dy_i - exp(y_i) * sum(dL/dy_j)
        for (int c = 0; c < C; c++) {
             atomicAdd(&current_dinp_row[c * stride_C], 
                       current_dout_row[c * stride_C] - expf(current_out_val_row[c * stride_C]) * sum_dout);
        }
    }
}

__global__ void mul_kernel(float* out, const float* a, const float* b, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

__global__ void mul_backward_kernel(float* da, float* db, const float* dout, const float* a_val, const float* b_val, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
         atomicAdd(&da[i], dout[i] * b_val[i]);
         atomicAdd(&db[i], dout[i] * a_val[i]);
    }
}

__global__ void mean_kernel(float* out, const float* inp, size_t n) {
    extern __shared__ float sdata[]; // Shared memory for reduction
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float mySum = 0.0f;
    // Each thread sums a portion of the input array
    while (i < n) {
        mySum += inp[i];
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block 0 writes the final sum for its block
    if (tid == 0) {
        atomicAdd(out, sdata[0]); // Add partial sum to the global output (must be initialized to 0)
    }
}

__global__ void divide_by_n_kernel(float* data, size_t n) {
    // Simple kernel assuming data[0] holds the sum
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n > 0) {
            data[0] = data[0] / (float)n;
        }
    }
}

__global__ void mean_backward_kernel(float* dinp, const float* dout, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && n > 0) {
        atomicAdd(&dinp[i], dout[0] / (float)n);
    }
}

__global__ void zero_kernel(float* data, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 0.0f;
    }
}

// --- Memory Management Implementation ---

void cuda_ensure_arr_on_gpu(Arr* a) {
    if (!a) return;
    if (!a->cuda_values) {
        // Allocate GPU memory
        CHECK_CUDA_ERROR(cudaMalloc((void**)&a->cuda_values, a->size * sizeof(float)));
        // Copy data from CPU if CPU data exists
        if (a->values) {
             CHECK_CUDA_ERROR(cudaMemcpy(a->cuda_values, a->values, a->size * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // If no CPU data, zero out GPU memory (or handle as error)
            CHECK_CUDA_ERROR(cudaMemset(a->cuda_values, 0, a->size * sizeof(float)));
        }
    }
}

void cuda_ensure_data_on_gpu(Tensor* t) {
    if (t && t->data) {
        cuda_ensure_arr_on_gpu(t->data);
    }
}

void cuda_ensure_grad_on_gpu(Tensor* t) {
     if (t && t->grad) {
        cuda_ensure_arr_on_gpu(t->grad);
    }
}

Arr* cuda_create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*)malloc(sizeof(Arr));
    if (!arr) { perror("malloc Arr failed"); exit(EXIT_FAILURE); }

    arr->ndim = ndim;
    arr->shape = (int*)malloc(ndim * sizeof(int));
    arr->strides = (int*)malloc(ndim * sizeof(int));
    if (!arr->shape || !arr->strides) { perror("malloc shape/strides failed"); free(arr); exit(EXIT_FAILURE); }
    memcpy(arr->shape, shape, ndim * sizeof(int));
    
    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }
    
    arr->values = NULL; // No CPU allocation by default for zeros
    arr->cuda_values = NULL;
    cuda_ensure_arr_on_gpu(arr); // Allocate and zero on GPU
    return arr;
}

Arr* cuda_create_arr(float* host_data, int* shape, int ndim) {
    Arr* arr = cuda_create_arr_zeros(shape, ndim); // Creates GPU array (zeroed)
    // Now copy host data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(arr->cuda_values, host_data, arr->size * sizeof(float), cudaMemcpyHostToDevice));
    // Optionally allocate and copy to CPU values as well if needed frequently
    // arr->values = (float*)malloc(arr->size * sizeof(float));
    // if(arr->values) memcpy(arr->values, host_data, arr->size * sizeof(float));
    return arr;
}

void cuda_free_arr(Arr* a) {
    if (a) {
        if (a->values) free(a->values); // Free CPU memory if it was allocated
        if (a->cuda_values) CHECK_CUDA_ERROR(cudaFree(a->cuda_values));
        free(a->shape);
        free(a->strides);
        free(a);
    }
}

Tensor* cuda_create_zero_tensor(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) { perror("malloc Tensor failed"); exit(EXIT_FAILURE); }
    t->data = cuda_create_arr_zeros(shape, ndim);
    t->grad = cuda_create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    // Ensure prevs are NULL pointers
    for(int i=0; i<MAX_PREVS; ++i) t->prevs[i] = NULL;
    return t;
}

Tensor* cuda_create_tensor(float* host_data, int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
     if (!t) { perror("malloc Tensor failed"); exit(EXIT_FAILURE); }
    t->data = cuda_create_arr(host_data, shape, ndim);
    t->grad = cuda_create_arr_zeros(shape, ndim); // Grad starts at zero
    t->op = -1;
    t->num_prevs = 0;
    for(int i=0; i<MAX_PREVS; ++i) t->prevs[i] = NULL;
    return t;
}

void cuda_free_tensor(Tensor* t) {
    if (t) {
        cuda_free_arr(t->data);
        cuda_free_arr(t->grad);
        // Do not free prevs, they are part of the graph structure
        free(t);
    }
}

void cuda_sync_tensor_data_to_host(Tensor* t) {
    if (!t || !t->data || !t->data->cuda_values) return;
    if (!t->data->values) { // Allocate host memory if it doesn't exist
        t->data->values = (float*)malloc(t->data->size * sizeof(float));
        if (!t->data->values) { perror("malloc host values failed"); exit(EXIT_FAILURE); }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(t->data->values, t->data->cuda_values, t->data->size * sizeof(float), cudaMemcpyDeviceToHost));
}

void cuda_sync_tensor_grad_to_host(Tensor* t) {
    if (!t || !t->grad || !t->grad->cuda_values) return;
     if (!t->grad->values) { // Allocate host memory if it doesn't exist
        t->grad->values = (float*)malloc(t->grad->size * sizeof(float));
        if (!t->grad->values) { perror("malloc host grad values failed"); exit(EXIT_FAILURE); }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(t->grad->values, t->grad->cuda_values, t->grad->size * sizeof(float), cudaMemcpyDeviceToHost));
}

void cuda_zero_grad(Tensor* t) {
    if (t && t->grad && t->grad->cuda_values) {
        // Option 1: Use cudaMemset (often efficient)
        // CHECK_CUDA_ERROR(cudaMemset(t->grad->cuda_values, 0, t->grad->size * sizeof(float)));

        // Option 2: Launch a kernel (can be better for many small tensors)
        int numBlocks = (t->grad->size + BLOCKSIZE - 1) / BLOCKSIZE;
        zero_kernel<<<numBlocks, BLOCKSIZE>>>(t->grad->cuda_values, t->grad->size);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

// --- CUDA Operations (Host functions launching kernels) Implementation ---

Tensor* cuda_matmul(Tensor* a, Tensor* b) {
    if (a->data->ndim != 2 || b->data->ndim != 2 || a->data->shape[1] != b->data->shape[0]) {
        fprintf(stderr, "CUDA Error: Incompatible shapes for matmul (%d,%d) x (%d,%d)\n", 
                a->data->shape[0], a->data->shape[1], b->data->shape[0], b->data->shape[1]);
        return NULL;
    }
    int M = a->data->shape[0];
    int K = a->data->shape[1];
    int N = b->data->shape[1];
    int shape[] = {M, N};
    Tensor* t = cuda_create_zero_tensor(shape, 2);
    
    cuda_ensure_data_on_gpu(a);
    cuda_ensure_data_on_gpu(b);

    // Simple block/grid configuration (adjust for performance)
    dim3 threadsPerBlock(16, 16); // Smaller block size might be okay
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
                   
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(a->data->cuda_values, b->data->cuda_values, t->data->cuda_values, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    t->op = MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* cuda_relu(Tensor* inp) {
    Tensor* t = cuda_create_zero_tensor(inp->data->shape, inp->data->ndim);
    cuda_ensure_data_on_gpu(inp);

    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    relu_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, inp->data->cuda_values, t->data->size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    t->op = RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

Tensor* cuda_logsoftmax(Tensor* inp) {
    if (inp->data->ndim != 2) {
         fprintf(stderr, "CUDA Error: logsoftmax expects input with ndim=2 (Batch, Classes)\n");
         return NULL;
    }
    Tensor* t = cuda_create_zero_tensor(inp->data->shape, inp->data->ndim);
    cuda_ensure_data_on_gpu(inp);

    int B = inp->data->shape[0];
    int C = inp->data->shape[1];
    int stride_B = inp->data->strides[0];
    int stride_C = inp->data->strides[1];
    
    int numBlocks = (B + BLOCKSIZE - 1) / BLOCKSIZE;
    logsoftmax_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, inp->data->cuda_values, B, C, stride_B, stride_C);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    t->op = LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

Tensor* cuda_mul(Tensor* a, Tensor* b) {
    if (a->data->size != b->data->size) {
        fprintf(stderr, "CUDA Error: Incompatible sizes for element-wise mul %d != %d\n", a->data->size, b->data->size);
        return NULL;
    } 
    Tensor* t = cuda_create_zero_tensor(a->data->shape, a->data->ndim);
    cuda_ensure_data_on_gpu(a);
    cuda_ensure_data_on_gpu(b);
    
    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mul_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, a->data->cuda_values, b->data->cuda_values, t->data->size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* cuda_mean(Tensor* inp) {
    int shape[] = {1};
    Tensor* m = cuda_create_zero_tensor(shape, 1); // Output tensor (size 1) is created zeroed on GPU
    cuda_ensure_data_on_gpu(inp);

    size_t n = inp->data->size;
    // --- Kernel Launch for Summation --- 
    // Determine grid size based on potential parallelism
    int maxThreadsPerBlock = BLOCKSIZE; // Or query device limits
    int numBlocks = (n + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    numBlocks = fminf(numBlocks, prop.multiProcessorCount * 4); // Heuristic: limit blocks

    // Shared memory size based on block size
    size_t sharedMemSize = maxThreadsPerBlock * sizeof(float); 

    // Launch sum kernel
    mean_kernel<<<numBlocks, maxThreadsPerBlock, sharedMemSize>>>(m->data->cuda_values, inp->data->cuda_values, n);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- Kernel Launch for Division --- 
    // Launch division kernel (only needs one thread)
    divide_by_n_kernel<<<1, 1>>>(m->data->cuda_values, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = inp;
    return m;
}

// --- CUDA Backward Operations (Host functions launching kernels) Implementation ---
void cuda_matmul_backward(Tensor* out) {
    if (!out || out->op != MATMUL || !out->prevs[0] || !out->prevs[1]) return;
    Tensor* a = out->prevs[0];
    Tensor* b = out->prevs[1];

    cuda_ensure_grad_on_gpu(out); // Ensure dC is on GPU
    cuda_ensure_data_on_gpu(a);   // Ensure A is on GPU
    cuda_ensure_data_on_gpu(b);   // Ensure B is on GPU
    cuda_ensure_grad_on_gpu(a);   // Ensure dA exists on GPU
    cuda_ensure_grad_on_gpu(b);   // Ensure dB exists on GPU

    int M = a->data->shape[0];
    int K = a->data->shape[1];
    int N = b->data->shape[1];

    dim3 threadsPerBlock(16, 16);
    
    // dA = dC @ B.T
    // Output dA is (M, K)
    dim3 numBlocksA((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_transpose_B_kernel<<<numBlocksA, threadsPerBlock>>>(
        out->grad->cuda_values, b->data->cuda_values, a->grad->cuda_values, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // dB = A.T @ dC
    // Output dB is (K, N)
    dim3 numBlocksB((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (K + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_A_matmul_kernel<<<numBlocksB, threadsPerBlock>>>(
        a->data->cuda_values, out->grad->cuda_values, b->grad->cuda_values, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_relu_backward(Tensor* out) {
     if (!out || out->op != RELU || !out->prevs[0]) return;
     Tensor* inp = out->prevs[0];
     
     cuda_ensure_grad_on_gpu(out); // Ensure dout is on GPU
     cuda_ensure_data_on_gpu(inp); // Ensure input value is on GPU
     cuda_ensure_grad_on_gpu(inp); // Ensure dinp exists on GPU

     int n = out->data->size;
     int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
     relu_backward_kernel<<<numBlocks, BLOCKSIZE>>>(inp->grad->cuda_values, out->grad->cuda_values, inp->data->cuda_values, n);
     CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_logsoftmax_backward(Tensor* out) {
     if (!out || out->op != LOGSOFTMAX || !out->prevs[0]) return;
     Tensor* inp = out->prevs[0];

     cuda_ensure_grad_on_gpu(out);       // Ensure dout is on GPU
     cuda_ensure_data_on_gpu(out);       // Ensure output value (logsoftmax val) is on GPU
     cuda_ensure_grad_on_gpu(inp);       // Ensure dinp exists on GPU

     int B = out->data->shape[0];
     int C = out->data->shape[1];
     int stride_B = inp->data->strides[0]; // Use input strides for dinp
     int stride_C = inp->data->strides[1];

     int numBlocks = (B + BLOCKSIZE - 1) / BLOCKSIZE;
     logsoftmax_backward_kernel<<<numBlocks, BLOCKSIZE>>>(
         inp->grad->cuda_values, out->grad->cuda_values, out->data->cuda_values, B, C, stride_B, stride_C);
     CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_mul_backward(Tensor* out) {
     if (!out || out->op != MUL || !out->prevs[0] || !out->prevs[1]) return;
     Tensor* a = out->prevs[0];
     Tensor* b = out->prevs[1];

     cuda_ensure_grad_on_gpu(out); // Ensure dout is on GPU
     cuda_ensure_data_on_gpu(a);   // Ensure a_val is on GPU
     cuda_ensure_data_on_gpu(b);   // Ensure b_val is on GPU
     cuda_ensure_grad_on_gpu(a);   // Ensure da exists on GPU
     cuda_ensure_grad_on_gpu(b);   // Ensure db exists on GPU

     int n = out->data->size;
     int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
     mul_backward_kernel<<<numBlocks, BLOCKSIZE>>>(
         a->grad->cuda_values, b->grad->cuda_values, out->grad->cuda_values, 
         a->data->cuda_values, b->data->cuda_values, n);
     CHECK_CUDA_ERROR(cudaGetLastError());
}

void cuda_mean_backward(Tensor* out) {
    if (!out || out->op != MEAN || !out->prevs[0]) return;
    Tensor* inp = out->prevs[0];

    cuda_ensure_grad_on_gpu(out); // Ensure dout (scalar) is on GPU
    cuda_ensure_grad_on_gpu(inp); // Ensure dinp exists on GPU

    size_t n = inp->data->size;
    int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    mean_backward_kernel<<<numBlocks, BLOCKSIZE>>>(inp->grad->cuda_values, out->grad->cuda_values, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// --- General Autograd Implementation ---
void cuda_backward(Tensor* t) {
     if (t == NULL || t->op == -1) { // Base case: leaf node or already processed
        return;
    }
    
    // Ensure gradient for current tensor is allocated on GPU
    // The initial call (e.g., on loss tensor) needs grad initialized to 1.0
    cuda_ensure_grad_on_gpu(t); 

    // Dispatch to specific CUDA backward function
    switch (t->op) {
        case MUL:
            cuda_mul_backward(t);
            break;
        case MEAN:
            cuda_mean_backward(t);
            break;
        case MATMUL:
            cuda_matmul_backward(t);
            break;
        case RELU:
            cuda_relu_backward(t);
            break;
        case LOGSOFTMAX:
            cuda_logsoftmax_backward(t);
            break;
        default:
            // fprintf(stderr, "Warning: CUDA backward pass not implemented for op %d\n", t->op);
            break;
    }

    // Recursively call backward on previous tensors
    // WARNING: Simple recursion can lead to recomputing gradients multiple times for shared nodes.
    // A proper implementation would use a topological sort or memoization.
    for (int i = 0; i < t->num_prevs; i++) {
         if (t->prevs[i] != NULL) {
            cuda_backward(t->prevs[i]);
         }
    }
}

// --- Debugging Implementation ---
void cuda_print_tensor(Tensor* t, const char* name) {
    if (!t) {
        printf("%s: NULL Tensor\n", name ? name : "Tensor");
        return;
    }
    printf("%s: Op=%d, Prevs=%d\n", name ? name : "Tensor", t->op, t->num_prevs);
    if (t->data) {
        cuda_sync_tensor_data_to_host(t); // Sync data GPU -> CPU before printing
        printf("  Data: ndim=%d, size=%d, shape=[%d", t->data->ndim, t->data->size, t->data->shape ? t->data->shape[0] : -1);
        for (int i = 1; i < t->data->ndim; i++) printf(",%d", t->data->shape[i]);
        printf("], values=[%f", t->data->values ? t->data->values[0] : NAN);
        int limit = t->data->size < 5 ? t->data->size : 5;
        for (int i = 1; i < limit; i++) printf(", %f", t->data->values[i]);
        if (t->data->size > limit) printf(", ...]"); else printf("]\n");
    } else {
        printf("  Data: NULL\n");
    }
     if (t->grad) {
        cuda_sync_tensor_grad_to_host(t); // Sync grad GPU -> CPU before printing
        printf("  Grad: ndim=%d, size=%d, shape=[%d", t->grad->ndim, t->grad->size, t->grad->shape ? t->grad->shape[0] : -1);
        for (int i = 1; i < t->grad->ndim; i++) printf(",%d", t->grad->shape[i]);
        printf("], values=[%f", t->grad->values ? t->grad->values[0] : NAN);
        int limit = t->grad->size < 5 ? t->grad->size : 5;
        for (int i = 1; i < limit; i++) printf(", %f", t->grad->values[i]);
        if (t->grad->size > limit) printf(", ...]"); else printf("]\n");
    } else {
         printf("  Grad: NULL\n");
    }
} 