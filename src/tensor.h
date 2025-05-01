#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef TENSOR_H
#define TENSOR_H

#define MAX_PREVS 3
#define MAX_ARGS 5
#define MAX_PARAM_TENSORS 10
// op codes
#define MATMUL 0
#define MEAN 1
#define MUL 2
#define RELU 3
#define LOGSOFTMAX 4

// Define M_PI if not already defined (e.g., on Windows with MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float* values;
    int* shape;
    int* strides;
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
    Arr* grad;
    int op; // op used to create this tensor
    struct Tensor* prevs[MAX_PREVS]; // tensors that were processed by the op
    int num_prevs;
    Arg args[MAX_ARGS]; // additional args for the op (e.g. axis, stride etc.)
} Tensor;

// Array operations
Arr* create_arr(float* data, int* shape, int ndim);
Arr* create_arr_zeros(int *shape, int ndim);
void free_arr(Arr* a);

// Tensor operations
Tensor* create_zero_tensor(int* shape, int ndim);
Tensor* create_tensor(float* data, int* shape, int ndim);
void free_tensor(Tensor* t);
void print_tensor(Tensor* t);

// Autograd
void backward(Tensor* t);

// Neural network operations (Forward)
Tensor* mul(Tensor* a, Tensor* b);
Tensor* mean(Tensor* a);
Tensor* matmul(Tensor* a, Tensor* b);
Tensor* logsoftmax(Tensor* inp);
Tensor* relu(Tensor* inp);

// Neural network operations (Backward)
void mul_backward(Tensor* out);
void mean_backward(Tensor* out);
void matmul_backward(Tensor* out);
void logsoftmax_backward(Tensor* out);
void relu_backward(Tensor* out);

// Initialization helpers
float random_normal();
float rand_float();
float rand_range(float min, float max);
float kaiming_uniform(int fan_in);
float kaiming_init(int fan_in);

#endif // TENSOR_H
