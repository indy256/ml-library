#include <stdio.h>
#include <tuple>

#define BLOCK_SIZE 16

typedef struct {
    int rows;
    int cols;
    int* m;
} Matrix;

__global__ void matrix_mul_kernel(Matrix d_a, Matrix d_b, Matrix d_c)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < d_a.rows && col < d_b.cols) 
    {
        int sum = 0;
        for(int i = 0; i < d_a.cols; i++) 
        {
            sum += d_a.m[row * d_a.cols + i] * d_b.m[i * d_b.cols + col];
        }
        d_c.m[row * d_b.cols + col] = sum;
    }
} 

std::tuple<Matrix, float> mul_gpu(const Matrix h_a, const Matrix h_b)
{
    int m = h_a.rows;
    int n = h_a.cols;
    int k = h_b.cols;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEventRecord(start, 0);

    Matrix d_a, d_b, d_c;
    cudaMalloc(&d_a.m, sizeof(int) * m * n);
    cudaMalloc(&d_b.m, sizeof(int) * n * k);
    cudaMalloc(&d_c.m, sizeof(int) * m * k);

    cudaMemcpy(d_a.m, h_a.m, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.m, h_b.m, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
   
    matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);    
    
    Matrix h_c { m, k };
    cudaMallocHost(&h_c.m, sizeof(int) * m * k);
    cudaMemcpy(h_c.m, d_c.m, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float duration_ms;
    cudaEventElapsedTime(&duration_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_a.m);
    cudaFree(d_b.m);
    cudaFree(d_c.m);
    cudaFreeHost(h_c.m);

    printf("Matrix %dx%dx%d multiplication on GPU took: %f ms\n\n", m, n, k, duration_ms);

    return { h_c, duration_ms };
}

Matrix cpu_matrix_transpose(const Matrix h_in) 
{
    Matrix h_out { h_in.cols, h_in.rows };
    cudaMallocHost(&h_out.m, sizeof(int) * h_in.rows * h_in.cols);

    for (int i = 0; i < h_in.rows; ++i) {
        for (int j = 0; j < h_in.cols; ++j) {
            h_out.m[j * h_in.rows + i] = h_in.m[i * h_in.cols + j];
        }
    }

    return h_out;
}

void cpu_matrix_mul(const Matrix h_a, const Matrix h_b, Matrix h_out) {
    Matrix h_bt = cpu_matrix_transpose(h_b);

    for (int i = 0; i < h_a.rows; i++) 
    {
        for (int j = 0; j < h_b.cols; j++) 
        {
            int sum = 0;
            for (int k = 0; k < h_a.cols; k++) 
            {
                sum += h_a.m[i * h_a.cols + k] * h_bt.m[j * h_a.cols + k];
                // sum += h_a.m[i * h_a.cols + k] * h_b.m[k * h_b.cols + j]; // 7x times slower
            }
            h_out.m[i * h_a.cols + j] = sum;
        }
    }

    cudaFreeHost(h_bt.m);
}

std::tuple<Matrix, float> mul_cpu(const Matrix h_a, const Matrix h_b)
{
    Matrix h_c;
    cudaMallocHost(&h_c.m, sizeof(int) * h_a.rows * h_b.cols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cpu_matrix_mul(h_a, h_b, h_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);              

    float duration_ms;
    cudaEventElapsedTime(&duration_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix %dx%dx%d multiplication on CPU took: %f ms\n\n", h_a.rows, h_a.cols, h_b.cols, duration_ms);

    return { h_c, duration_ms };
}


int main(int argc, char const *argv[])
{
    int m = 1000;
    int n = 1000;
    int k = 1000;

    Matrix h_a { m, n };
    Matrix h_b { n, k };

    cudaMallocHost(&h_a.m, sizeof(int) * h_a.rows * h_a.cols);
    cudaMallocHost(&h_b.m, sizeof(int) * h_b.rows * h_b.cols);

    srand(1);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a.m[i * n + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b.m[i * k + j] = rand() % 1024;
        }
    }

    Matrix h_c_gpu, h_c_cpu;
    float gpu_duration_ms, cpu_duration_ms;
    std::tie (h_c_gpu, gpu_duration_ms) = mul_gpu(h_a, h_b);
    std::tie (h_c_cpu, cpu_duration_ms) = mul_cpu(h_a, h_b);

    bool ok = true;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            ok &= h_c_gpu.m[i * k + j] == h_c_cpu.m[i * k + j];
        }
    }

    if (ok)
    {
        printf("GPU speedup = %f\n", cpu_duration_ms / gpu_duration_ms);
    }
    else
    {
        printf("GPU and CPU results don't match\n");
    }

    cudaFreeHost(h_a.m);
    cudaFreeHost(h_b.m);
    cudaFreeHost(h_c_gpu.m);
    cudaFreeHost(h_c_cpu.m);
    return 0;
}
