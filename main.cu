#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Fonctions CPU
void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);

// Fonctions GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n);

int main() {
    int n = 1000; // Taille de la matrice (n x n)
    size_t size = n * n * sizeof(float);

    // Allocation mémoire sur CPU
    float *h_M1 = (float *)malloc(size);
    float *h_M2 = (float *)malloc(size);
    float *h_Mout = (float *)malloc(size);

    // Initialisation des matrices
    MatrixInit(h_M1, n, n);
    MatrixInit(h_M2, n, n);

    // Affichage des matrices
    printf("Matrice 1 (partielle) :\n");
    MatrixPrint(h_M1, 4, 4); // Affiche une partie de la matrice
    printf("Matrice 2 (partielle) :\n");
    MatrixPrint(h_M2, 4, 4); // Affiche une partie de la matrice

    // Mesurer le temps d'addition sur CPU
    clock_t start = clock();
    MatrixAdd(h_M1, h_M2, h_Mout, n, n);
    clock_t end = clock();
    double cpu_time_add = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps d'execution de l'addition sur CPU : %f secondes\n", cpu_time_add);

    // Mesurer la multiplication sur CPU
    start = clock();
    MatrixMult(h_M1, h_M2, h_Mout, n);
    end = clock();
    double cpu_time_mult = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps d'execution de la multiplication sur CPU : %f secondes\n", cpu_time_mult);

    // Allocation mémoire sur GPU
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void **)&d_M1, size);
    cudaMalloc((void **)&d_M2, size);
    cudaMalloc((void **)&d_Mout, size);

    // Transfert des données CPU -> GPU
    cudaMemcpy(d_M1, h_M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, size, cudaMemcpyHostToDevice);

    // Configuration des dimensions de grid et block
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    // Mesurer le temps d'addition sur GPU
    cudaEvent_t start_gpu, stop_gpu;
    float elapsedTime;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);
    cudaMatrixAdd<<<numBlocks, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsedTime, start_gpu, stop_gpu);
    printf("Temps d'execution de l'addition sur GPU : %f ms\n", elapsedTime);

    // Mesurer la multiplication sur GPU
    cudaEventRecord(start_gpu);
    cudaMatrixMult<<<numBlocks, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsedTime, start_gpu, stop_gpu);
    printf("Temps d'execution de la multiplication sur GPU : %f ms\n", elapsedTime);

    // Calcul de l'accélération réelle
    double accel_add_real = cpu_time_add / (elapsedTime / 1000);  // Convertir GPU en secondes
    double accel_mult_real = cpu_time_mult / (elapsedTime / 1000);
    printf("Acceleration reelle pour l'addition : %f\n", accel_add_real);
    printf("Acceleration reelle pour la multiplication : %f\n", accel_mult_real);

    // Libération de la mémoire
    free(h_M1);
    free(h_M2);
    free(h_Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}

// Fonction pour initialiser une matrice avec des valeurs aléatoires entre -1 et 1
void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        M[i] = (float)(rand() % 200 - 100) / 100.0;
    }
}

// Fonction pour afficher une matrice
void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// Fonction pour additionner deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n * p; i++) {
        Mout[i] = M1[i] + M2[i];
    }
}

// Fonction pour multiplier deux matrices sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

// Kernel pour additionner deux matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        int index = row * p + col;
        Mout[index] = M1[index] + M2[index];
    }
}

// Kernel pour multiplier deux matrices sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = sum;
    }
}
