#include "tensor.h"

// Array operations Implementation
Arr* create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*) malloc(sizeof(Arr));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->shape = (int*) malloc(ndim * sizeof(int));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(int));

    arr->strides = (int*) malloc(ndim * sizeof(int));
    if (!arr->strides) {
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }

    arr->values = (float*) calloc(arr->size, sizeof(float));
    if (!arr->values) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    return arr;
}

Arr* create_arr(float* data, int* shape, int ndim) {
    Arr* arr = create_arr_zeros(shape, ndim);
    if (!arr) return NULL; // Check if create_arr_zeros failed
    memcpy(arr->values, data, arr->size * sizeof(float));
    return arr;
}

void free_arr(Arr* a) {
    if (a == NULL) return;
    if (a->values != NULL) {
        free(a->values);
    }
    if (a->shape != NULL) {
        free(a->shape);
    }
    if (a->strides != NULL) {
        free(a->strides);
    }
    free(a);
}

// Tensor operations Implementation
Tensor* create_tensor(float* data, int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->data = create_arr(data, shape, ndim);
    if (!t->data) {
        free(t);
        return NULL;
    }
    t->grad = create_arr_zeros(shape, ndim);
    if (!t->grad) {
        free_arr(t->data);
        free(t);
        return NULL;
    }
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

Tensor* create_zero_tensor(int* shape, int ndim) {
     Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->data = create_arr_zeros(shape, ndim);
     if (!t->data) {
        free(t);
        return NULL;
    }
    t->grad = create_arr_zeros(shape, ndim);
    if (!t->grad) {
        free_arr(t->data);
        free(t);
        return NULL;
    }
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

void free_tensor(Tensor* t) {
    if (t == NULL) return;
    // Note: Gradients are usually part of the overall computation graph 
    // and might be freed separately or managed by a graph context.
    // Simple freeing here might lead to double-free if grad points to shared data.
    // However, assuming independent allocation for now:
    if (t->data != NULL) free_arr(t->data);
    if (t->grad != NULL) free_arr(t->grad);
    // We don't free t->prevs tensors here as they are managed elsewhere (part of the graph)
    free(t);
}

void print_tensor(Tensor* t) {
    printf("Tensor(\n");
    if (t->data) {
        printf("\tdata(%d): ", t->data->size);
        // Print limited number of elements for large tensors
        int limit = t->data->size < 10 ? t->data->size : 10;
        for (int i = 0; i < limit; i++) printf("%f,", t->data->values[i]);
        if (t->data->size > limit) printf("...");
        printf("\n\tshape: ");
        for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->shape[i]);
        printf("\n\tstrides: ");
        for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->strides[i]);
    } else {
        printf("\tdata: NULL\n");
    }
     if (t->grad) {
        printf("\n\tgrad(%d): ", t->grad->size);
        int limit = t->grad->size < 10 ? t->grad->size : 10;
        for (int i = 0; i < limit; i++) printf("%f,", t->grad->values[i]);
         if (t->grad->size > limit) printf("...");
    } else {
        printf("\n\tgrad: NULL\n");
    }
    printf("\n\top: %d, num_prevs: %d", t->op, t->num_prevs);
    printf("\n)\n");
}


// Autograd Implementation
void backward(Tensor* t) {
    if (t == NULL || t->op == -1) { // Base case: No operation created this tensor, or already visited/no grad needed
        return;
    }

    // Specific backward functions compute gradients for `t->prevs` based on `t->grad`
    switch (t->op) {
        case MUL:
            mul_backward(t);
            break;
        case MEAN:
            mean_backward(t);
            break;
        case MATMUL:
            matmul_backward(t);
            break;
        case RELU:
            relu_backward(t);
            break;
        case LOGSOFTMAX:
            logsoftmax_backward(t);
            break;
        // Add cases for other operations here
        default:
            // Optionally handle unknown ops or ops without backward pass
            break;
    }

    // Recursively call backward on previous tensors
    // Basic recursion; might need optimization (e.g., topological sort) for complex graphs
    for (int i = 0; i < t->num_prevs; i++) {
         if (t->prevs[i] != NULL) { // Check if prev tensor exists
            backward(t->prevs[i]);
         }
    }
}

// Neural network operations (Forward) Implementation
Tensor* mul(Tensor* a, Tensor* b) {
    // Basic check: Ensure shapes match for element-wise multiplication
    if (a->data->size != b->data->size) return NULL; // Or handle broadcasting
    Tensor* t = create_zero_tensor(a->data->shape, a->data->ndim);
    if (!t) return NULL;

    #pragma omp parallel for
    for (int i = 0; i < a->data->size; i++) {
        t->data->values[i] = a->data->values[i] * b->data->values[i];
    }
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* mean(Tensor* inp) {
    Tensor* m = create_zero_tensor((int[]){1}, 1);
    if (!m) return NULL;
    float s = 0.0f;
    // Consider using Kahan summation for better precision with large sums
    #pragma omp parallel for reduction(+:s)
    for(int i = 0; i < inp->data->size; i++) s += inp->data->values[i];
    m->data->values[0] = s / inp->data->size;
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = inp;
    return m;
}

Tensor* matmul(Tensor* a, Tensor* b) {
    // Basic check: Ensure dimensions are compatible for matmul
    // (P,Q) x (Q,R) = (P,R)
    if (a->data->ndim != 2 || b->data->ndim != 2 || a->data->shape[1] != b->data->shape[0]) {
        fprintf(stderr, "Error: Incompatible shapes for matmul\n");
        return NULL;
    }
    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];
    Tensor* t = create_zero_tensor((int[]) {P, R}, 2);
     if (!t) return NULL;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < Q; k++) {
                // Using strides for potentially non-contiguous memory access
                int pos_a = i * a->data->strides[0] + k * a->data->strides[1];
                int pos_b = k * b->data->strides[0] + j * b->data->strides[1];
                tmp += a->data->values[pos_a] * b->data->values[pos_b];
            }
            // Output strides are simple for a new contiguous tensor
            int pos_c = i * t->data->strides[0] + j * t->data->strides[1]; 
            t->data->values[pos_c] = tmp;
        }
    }
    t->op = MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* logsoftmax(Tensor* inp) {
    // Assumes inp shape (B, C) where B is batch size, C is number of classes
    if (inp->data->ndim != 2) {
        fprintf(stderr, "Error: logsoftmax expects input with ndim=2 (Batch, Classes)\n");
        return NULL;
    }
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    if (!t) return NULL;
    int B = inp->data->shape[0];
    int C = inp->data->shape[1];

    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        // Find max value for numerical stability
        float maxv = -INFINITY;
        for (int c = 0; c < C; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            if (inp->data->values[pos] > maxv) {
                maxv = inp->data->values[pos];
            }
        }

        // Calculate sum of exps
        float sumexp = 0.0f;
        for (int c = 0; c < C; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            sumexp += expf(inp->data->values[pos] - maxv);
        }
        float logsumexp = logf(sumexp);

        // Calculate logsoftmax
        for (int c = 0; c < C; c++) {
            int pos = b * inp->data->strides[0] + c * inp->data->strides[1];
            int out_pos = b * t->data->strides[0] + c * t->data->strides[1];
            t->data->values[out_pos] = inp->data->values[pos] - maxv - logsumexp;
        }
    }
    t->op = LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

Tensor* relu(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    if (!t) return NULL;
    #pragma omp parallel for
    for (int i = 0; i < inp->data->size; i++) {
        t->data->values[i] = (inp->data->values[i] > 0) ? inp->data->values[i] : 0.0f;
    }
    t->op = RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

// Neural network operations (Backward) Implementation
void mul_backward(Tensor* out) {
    if (!out || !out->prevs[0] || !out->prevs[1] || !out->grad) return;
    Arr* da = out->prevs[0]->grad;
    Arr* db = out->prevs[1]->grad;
    Arr* dout = out->grad;
    Arr* val_a = out->prevs[0]->data;
    Arr* val_b = out->prevs[1]->data;

    #pragma omp parallel for
    for (int i = 0; i < out->data->size; i++) {
        // Check for NULL grad arrays if they might not be initialized
        if (da) da->values[i] += dout->values[i] * val_b->values[i];
        if (db) db->values[i] += dout->values[i] * val_a->values[i];
    }
}

void mean_backward(Tensor* out) {
    if (!out || !out->prevs[0] || !out->grad) return;
    Arr* dinp = out->prevs[0]->grad;
    Arr* dout = out->grad;
    int size = out->prevs[0]->data->size;
    float grad_val = dout->values[0] / size;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
         if (dinp) dinp->values[i] += grad_val;
    }
}

void matmul_backward(Tensor* out) {
    if (!out || !out->prevs[0] || !out->prevs[1] || !out->grad) return;
    Tensor* a = out->prevs[0];
    Tensor* b = out->prevs[1];
    Arr* da = a->grad;
    Arr* db = b->grad;
    Arr* dout = out->grad;

    if (!da || !db) return; // Ensure grad arrays exist

    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];
    
    // dA = dC @ B.T  (P,R) x (R,Q) => (P,Q)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < R; k++) {
                int pos_dout = i * dout->strides[0] + k * dout->strides[1];
                // (k,j) in b.T is (j,k) in b
                int pos_b = j * b->data->strides[0] + k * b->data->strides[1]; 
                tmp += dout->values[pos_dout] * b->data->values[pos_b];
            }
            int pos_da = i * da->strides[0] + j * da->strides[1];
            da->values[pos_da] += tmp; // Use += to accumulate gradients
        }
    }
    
    // dB = A.T @ dC  (Q,P) x (P,R) => (Q,R)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Q; i++) {
        for (int j = 0; j < R; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < P; k++) {
                // (i,k) in a.T is (k,i) in a
                int pos_a = k * a->data->strides[0] + i * a->data->strides[1]; 
                int pos_dout = k * dout->strides[0] + j * dout->strides[1];
                tmp += a->data->values[pos_a] * dout->values[pos_dout];
            }
            int pos_db = i * db->strides[0] + j * db->strides[1];
            db->values[pos_db] += tmp; // Use += to accumulate gradients
        }
    }   
}

void logsoftmax_backward(Tensor* out) {
    if (!out || !out->prevs[0] || !out->grad) return;
    // out shape (B,C), prevs[0] shape (B,C)
    Tensor* inp = out->prevs[0];
    Arr* dinp = inp->grad;
    Arr* dout = out->grad;
    Arr* val_out = out->data; // logsoftmax output values

    if (!dinp) return;

    int B = out->data->shape[0];
    int C = out->data->shape[1];

    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        float gradsum = 0.0f;
        for (int c = 0; c < C; c++) {
            int pos = b * dout->strides[0] + c * dout->strides[1];
            gradsum += dout->values[pos];
        }
        for (int c = 0; c < C; c++) {
            int pos = b * dinp->strides[0] + c * dinp->strides[1];
            int pos_dout = b * dout->strides[0] + c * dout->strides[1];
            int pos_val_out = b * val_out->strides[0] + c * val_out->strides[1];
            // dL/dx_i = dL/dy_i - exp(y_i) * sum(dL/dy_j)
            dinp->values[pos] += dout->values[pos_dout] - expf(val_out->values[pos_val_out]) * gradsum;
        }
    }
}

void relu_backward(Tensor* out) {
    if (!out || !out->prevs[0] || !out->grad) return;
    Tensor* inp = out->prevs[0];
    Arr* dinp = inp->grad;
    Arr* dout = out->grad;
    Arr* val_inp = inp->data;

    if (!dinp) return;

    #pragma omp parallel for
    for (int i = 0; i < out->data->size; i++) {
        dinp->values[i] += (val_inp->values[i] > 0) ? dout->values[i] : 0.0f;
    }
}

// Initialization helpers Implementation
float random_normal() {
    // Box-Muller transform
    float u1 = rand_float(); // want range (0, 1]
    float u2 = rand_float();
    // Ensure u1 is not exactly 0 to avoid log(0)
    while (u1 == 0.0f) {
        u1 = rand_float();
    }
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

float rand_float() {
    // RAND_MAX + 1 might overflow, be careful. Cast to double for intermediate.
    return (float)rand() / ((double)RAND_MAX + 1.0);
}

float rand_range(float min, float max) {
    return min + rand_float() * (max - min);
}

float kaiming_uniform(int fan_in) {
    if (fan_in <= 0) fan_in = 1; // Avoid division by zero or sqrt of zero
    float gain = sqrtf(2.0f);  // Standard gain for ReLU
    float std = gain / sqrtf((float)fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
}

float kaiming_init(int fan_in) {
     if (fan_in <= 0) fan_in = 1; // Avoid division by zero
    float std_dev = sqrtf(2.0f / (float)fan_in);
    return random_normal() * std_dev;
} 