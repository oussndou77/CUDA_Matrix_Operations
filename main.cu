#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Prototypes des fonctions CPU et GPU
void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);
void subsampling2D(float *input, float *output, int input_size, int output_size, int num_channels);
void convolution2D(float *input, float *output, float *kernels, int input_size, int kernel_size, int output_size, int num_kernels);

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n);
// Kernel pour effectuer la convolution 2D avec activation tanh
__global__ void cudaConvolution2D(float *input, float *output, float *kernels, int input_size, int kernel_size, int output_size, int num_kernels);


__device__ float activation_tanh(float M);

// Initialisation des matrices spécifiques à LeNet-5
void init_matrix(float *matrix, int size);
void init_zero(float *matrix, int size);

int main(int argc, char *argv[]) {
    // Vérifier les arguments
    if (argc != 2) {
        printf("Usage : %s <taille de la matrice (n x n)>\n", argv[0]);
        return -1;
    }

    // Lire la taille de la matrice à partir des arguments
    int n = atoi(argv[1]); // Convertir l'argument en entier
    if (n <= 0) {
        printf("Erreur : La taille de la matrice doit être un entier positif.\n");
        return -1;
    }

    // Taille de la mémoire à allouer pour chaque matrice
    size_t size = n * n * sizeof(float);

    // Allocation mémoire sur CPU
    float *h_M1 = (float *)malloc(size);
    float *h_M2 = (float *)malloc(size);
    float *h_Mout = (float *)malloc(size);

    // Initialisation des matrices
    MatrixInit(h_M1, n, n);
    MatrixInit(h_M2, n, n);

    printf("Matrice 1 (partielle) :\n");
    MatrixPrint(h_M1, 4, 4); // Affiche une portion de la matrice
    printf("Matrice 2 (partielle) :\n");
    MatrixPrint(h_M2, 4, 4);

    // Partie 2 - Matrices LeNet-5
    printf("\nInitialisation des matrices pour LeNet-5\n");

    // Dimensions
    int raw_size = 32;          // Taille d'entrée
    int kernel_size = 5;        // Taille du noyau
    int C1_size = 28;           // Taille de sortie après convolution
    int S1_size = 14;           // Taille de sortie après sous-échantillonnage
    int num_kernels = 6;        // Nombre de noyaux

    // Allocation mémoire
    float *raw_data = (float *)malloc(raw_size * raw_size * sizeof(float));
    float *C1_data = (float *)malloc(num_kernels * C1_size * C1_size * sizeof(float));
    float *C1_data_woaf = (float *)malloc(num_kernels * C1_size * C1_size * sizeof(float));
    float *S1_data = (float *)malloc(num_kernels * S1_size * S1_size * sizeof(float));
    float *C1_kernel = (float *)malloc(num_kernels * kernel_size * kernel_size * sizeof(float));

    // Allocation sur GPU
    float *d_input, *d_output, *d_kernels;
    cudaMalloc((void **)&d_input, raw_size * raw_size * sizeof(float));
    cudaMalloc((void **)&d_output, num_kernels * C1_size * C1_size * sizeof(float));
    cudaMalloc((void **)&d_kernels, num_kernels * kernel_size * kernel_size * sizeof(float));

    // Copier les données vers le GPU
    cudaMemcpy(d_input, raw_data, raw_size * raw_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, C1_kernel, num_kernels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Configuration des blocs et grilles
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((C1_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (C1_size + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   num_kernels);

    // Lancer le kernel de convolution avec activation tanh
    cudaConvolution2D<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernels, raw_size, kernel_size, C1_size, num_kernels);

    // Copier les résultats du GPU vers le CPU
    cudaMemcpy(C1_data, d_output, num_kernels * C1_size * C1_size * sizeof(float), cudaMemcpyDeviceToHost);



    // Initialisation des matrices
    init_matrix(raw_data, raw_size);   // raw_data entre 0 et 1
    init_zero(C1_data_woaf, C1_size);  // C1_datawoaf initialisé à 0
    init_zero(C1_data, C1_size);    // C1_data initialisé à 0
    init_zero(S1_data, S1_size);       // S1_data initialisé à 0
    init_matrix(C1_kernel, kernel_size); // C1_kernel entre 0 et 1

    // Afficher quelques valeurs de raw_data
    printf("raw_data (4 premiers elements) : ");
    for (int i = 0; i < 4; i++) printf("%.2f ", raw_data[i]);
    printf("\n");

    // Convolution 2D
    convolution2D(raw_data, C1_data_woaf, C1_kernel, raw_size, kernel_size, C1_size, num_kernels);


    // Afficher quelques valeurs de C1_data sans fonction d'activation
    printf("C1_data_woaf sans fonction d'activation tanh (4 premiers elements) : ");
    for (int i = 0; i < 4; i++) printf("%.2f ", C1_data_woaf[i]);
    printf("\n");
    
    // Afficher quelques valeurs de C1_data avec la fonction d'activation tanh
    printf("C1_data avec la fonction d'activation tanh (4 premiers elements) : ");
    for (int i = 0; i < 4; i++) {
        printf("%.2f ", C1_data[i]);
    }
    printf("\n");

    // Sous-échantillonnage 2D
    subsampling2D(C1_data, S1_data, C1_size, S1_size, num_kernels);

    // Afficher quelques valeurs de S1_data
    printf("S1_data (4 premiers elements) : ");
    for (int i = 0; i < 4; i++) printf("%.2f ", S1_data[i]);
    printf("\n");

    // Libération mémoire pour LeNet-5
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);


    // Addition sur CPU
    clock_t start = clock();
    MatrixAdd(h_M1, h_M2, h_Mout, n, n);
    clock_t end = clock();
    double cpu_time_add = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps d'execution de l'addition sur CPU : %f secondes\n", cpu_time_add);

    // Multiplication sur CPU
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

    

    // Addition sur GPU
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

    // Multiplication sur GPU
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


    // Libération mémoire
    free(h_M1);
    free(h_M2);
    free(h_Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}

// Fonction d'initialisation pour les matrices entre 0 et 1
void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX; // Valeurs entre 0 et 1
    }
}

// Fonction d'initialisation à 0
void init_zero(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f;
    }
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

// Fonction pour effectuer la convolution 2D
void convolution2D(float *input, float *output, float *kernels, int input_size, int kernel_size, int output_size, int num_kernels) {
    for (int k = 0; k < num_kernels; k++) { // Pour chaque noyau
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        int input_row = i + ki;
                        int input_col = j + kj;
                        sum += input[input_row * input_size + input_col] *
                               kernels[k * kernel_size * kernel_size + ki * kernel_size + kj];
                    }
                }
                output[k * output_size * output_size + i * output_size + j] = sum;
            }
        }
    }
}

// Fonction pour effectuer le sous-échantillonnage 2D (moyennage)
void subsampling2D(float *input, float *output, int input_size, int output_size, int num_channels) {
    for (int c = 0; c < num_channels; c++) { // Pour chaque canal
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                int input_row = i * 2;
                int input_col = j * 2;
                float sum = 0.0f;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        sum += input[c * input_size * input_size + (input_row + ki) * input_size + (input_col + kj)];
                    }
                }
                output[c * output_size * output_size + i * output_size + j] = sum / 4.0f;
            }
        }
    }
}

// Fonction d'activation tanh
__device__ float activation_tanh(float M) {
    return tanhf(M); // Fonction hyperbolique tangente optimisée
}

// Kernel pour effectuer la convolution 2D avec activation tanh
__global__ void cudaConvolution2D(float *input, float *output, float *kernels, int input_size, int kernel_size, int output_size, int num_kernels) {
    int k = blockIdx.z; // Identifiant du noyau (canal)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Ligne de la matrice de sortie
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Colonne de la matrice de sortie

    if (k < num_kernels && i < output_size && j < output_size) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernel_size; ki++) {
            for (int kj = 0; kj < kernel_size; kj++) {
                int input_row = i + ki;
                int input_col = j + kj;
                sum += input[input_row * input_size + input_col] * kernels[k * kernel_size * kernel_size + ki * kernel_size + kj];
            }
        }
        // Appliquer l'activation tanh sur la somme calculée
        output[k * output_size * output_size + i * output_size + j] = activation_tanh(sum);
    }
}


// Pour la fonction d'activation softmax :
// 4. Softmax Kernel
__global__ void softmax(float* input, float* output, int size) {
    extern __shared__ float temp[];
    int idx = threadIdx.x;
    if (idx < size) {
        temp[idx] = expf(input[idx]);
        __syncthreads();
        float sum = 0.0f;
        for (int i = 0; i < size; i++) sum += temp[i];
        output[idx] = temp[idx] / sum;
    }
}
// Pour le   keras.layers.Dense(120, activation='tanh'), #C5
// 5. Fully Connected Layer Kernel for Dense 120
__global__ void fullyConnected120(float* input, float* weights, float* biases, float* output, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 120) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = tanhf(sum);
    }
}

// Pour le   keras.layers.Dense(84, activation='tanh'), #C5
// 6. Fully Connected Layer Kernel for Dense 84
__global__ void fullyConnected84(float* input, float* weights, float* biases, float* output, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 84) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = tanhf(sum);
    }
}

// Importation des poids dans les matrices CUDA
void loadWeights(const char *file_path, float *host_data, int size) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        printf("Erreur : impossible d'ouvrir le fichier %s\n", file_path);
        exit(1);
    }
    fread(host_data, sizeof(float), size, file);
    fclose(file);
}

