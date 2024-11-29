#include <stdio.h>
#include <stdlib.h>
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
    int n = 1000; // Taille des matrices carrées
    size_t size = n * n * sizeof(float);

    // Allocation mémoire sur CPU
    float *h_M1 = (float *)malloc(size);
    float *h_M2 = (float *)malloc(size);
    float *h_Mout = (float *)malloc(size);

    // Initialisation des matrices
    MatrixInit(h_M1, n, n);
    MatrixInit(h_M2, n, n);

    printf("Matrice 1 :\n");
    MatrixPrint(h_M1, n, n);

    printf("Matrice 2 :\n");
    MatrixPrint(h_M2, n, n);

    // Addition sur CPU
    MatrixAdd(h_M1, h_M2, h_Mout, n, n);
    printf("Resultat de l'addition (CPU) :\n");
    MatrixPrint(h_Mout, n, n);

    // Multiplication sur CPU
    MatrixMult(h_M1, h_M2, h_Mout, n);
    printf("Resultat de la multiplication (CPU) :\n");
    MatrixPrint(h_Mout, n, n);

    // Allocation mémoire sur GPU
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void **)&d_M1, size);
    cudaMalloc((void **)&d_M2, size);
    cudaMalloc((void **)&d_Mout, size);

    // Transfert des données CPU -> GPU
    cudaMemcpy(d_M1, h_M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, size, cudaMemcpyHostToDevice);

    // Configuration des dimensions du grid et du block
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Addition sur GPU
    cudaMatrixAdd<<<numBlocks, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, n);
    cudaMemcpy(h_Mout, d_Mout, size, cudaMemcpyDeviceToHost);
    printf("Resultat de l'addition (GPU) :\n");
    MatrixPrint(h_Mout, n, n);

    // Multiplication sur GPU
    cudaMatrixMult<<<numBlocks, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n);
    // Synchronisation et transfert des résultats GPU -> CPU
    cudaMemcpy(h_Mout, d_Mout, size, cudaMemcpyDeviceToHost);
    // Affichage des résultats
    printf("Resultat de la multiplication (GPU) :\n");
    MatrixPrint(h_Mout, n, n);

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
    printf("\n");
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
