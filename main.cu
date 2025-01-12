/****************************************************/
/*                 main.cu complet                 */
/*       Lecture MNIST + Affichage console +        */
/*         Inférence LeNet-5 + Prédiction           */
/****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

//==================================================//
//  Constantes / dimensions pour MNIST et LeNet-5   //
//==================================================//
static const int MNIST_ROWS   = 28;
static const int MNIST_COLS   = 28;
static const int RAW_SIZE     = 32; // padding 2 sur chaque bord (28 → 32)
static const int KERNEL_SIZE  = 5;

static const int C1_OUT_CH   = 6;
static const int C1_OUT_SIZE = 28;  // 32 - 5 + 1

static const int S1_OUT_SIZE = 14;  // pooling 2x2

static const int C3_OUT_CH   = 16;
static const int C3_OUT_SIZE = 10;  // 14 - 5 + 1

static const int S4_OUT_SIZE = 5;   // pooling 2x2 -> 5

// fully-connected
static const int FC1_SIZE = 120;  // layer 5
static const int FC2_SIZE = 84;   // layer 6
static const int FC3_SIZE = 10;   // layer 7 (sortie)

//==================================================//
//   Prototypes pour lecture/affichage MNIST        //
//==================================================//
unsigned int swapEndian(unsigned int value);
float* readMNISTImage(const char* filename, int imgIndex);

// Pour l'affichage en console
void charBckgrndPrint(char *str, int rgb[3]);
void imgColorPrint(int height, int width, int ***img);
int ***allocImgRGB(int height, int width);
void freeImgRGB(int ***img, int height, int width);

//==================================================//
//   Prototypes pour chargement de poids / biais    //
//==================================================//
void loadWeights(const char *file_path, float *host_data, int size);

//==================================================//
//   Prototypes pour la convolution multi-canal     //
//==================================================//
__global__ void cudaConv2DMultiChannel(
    float *input,    // [in_channels, input_size, input_size]
    float *output,   // [out_channels, output_size, output_size]
    float *kernels,  // [out_channels, in_channels, 5,5]
    float *bias,     // [out_channels]
    int in_channels,
    int input_size,
    int kernel_size,
    int out_channels,
    int output_size
);

// Pooling (CPU)
void subsampling2D(float* input, float* output, 
                   int input_size, int output_size,
                   int num_channels);

// Flatten (CPU)
void flattenCPU(float* input, float* output, 
                int channels, int height, int width);

//==================================================//
//   Fully-Connected (GPU) + Softmax                //
//==================================================//
__global__ void fullyConnected120(float* input, float* weights, float* biases,
                                  float* output, int input_size);
__global__ void fullyConnected84(float* input, float* weights, float* biases,
                                 float* output, int input_size);
__global__ void fullyConnected10(float* input, float* weights, float* biases,
                                 float* output, int input_size);

__global__ void softmaxKernel(float* input, float* output, int size);

// argmax CPU
int argmaxCPU(float* data, int size);

//==================================================//
//                   main()                         //
//==================================================//
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage : %s <fichier_mnist_images> <index_image>\n", argv[0]);
        return 1;
    }
    const char* mnistFile = argv[1];
    int imgIndex = atoi(argv[2]);

    /*****************************************************/
    /*     1) Lire l'image MNIST (28x28) et l'afficher   */
    /*****************************************************/
    float* mnistImage = readMNISTImage(mnistFile, imgIndex);

    // On crée un tableau [28][28][3] pour l'affichage "couleur"
    int ***imgRGB = allocImgRGB(MNIST_ROWS, MNIST_COLS);

    // Remplir en nuances de gris inversées
    //  - plus la valeur est élevée, plus on met une teinte foncée
    for(int i=0; i<MNIST_ROWS; i++){
        for(int j=0; j<MNIST_COLS; j++){
            float val = mnistImage[i*MNIST_COLS + j]; // [0..1]
            int intensity = (int)(255 - (val * 255)); // inversé
            imgRGB[i][j][0] = intensity;
            imgRGB[i][j][1] = intensity;
            imgRGB[i][j][2] = intensity;
        }
    }

    // Afficher dans la console
    printf("=== IMAGE MNIST index %d ===\n", imgIndex);
    imgColorPrint(MNIST_ROWS, MNIST_COLS, imgRGB);
    // on peut libérer la structure d'affichage
    freeImgRGB(imgRGB, MNIST_ROWS, MNIST_COLS);

    // A ce stade, on sait exactement à quoi ressemble l'image traitée
    // On passe maintenant à l'inférence LeNet-5

    /*****************************************************/
    /* 2) Préparer l'input 32x32 (padding) + GPU        */
    /*****************************************************/
    float* d_inputImage;
    cudaMallocManaged(&d_inputImage, RAW_SIZE * RAW_SIZE * sizeof(float));
    // Mettre tout à 0
    for(int i=0; i<RAW_SIZE*RAW_SIZE; i++){
        d_inputImage[i] = 0.0f;
    }
    // Copier l'image 28x28 au centre (offset=2)
    for(int r=0; r<28; r++){
        for(int c=0; c<28; c++){
            d_inputImage[(r+2)*RAW_SIZE + (c+2)] = mnistImage[r*28 + c];
        }
    }
    free(mnistImage);

    /*****************************************************/
    /* 3) Charger les poids et biais depuis *.bin        */
    /*****************************************************/
    // C1
    float *d_C1_kernels, *d_C1_bias;
    cudaMallocManaged(&d_C1_kernels, C1_OUT_CH * 1 * KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMallocManaged(&d_C1_bias,    C1_OUT_CH * sizeof(float));
    loadWeights("weights_layer_0.bin", d_C1_kernels, C1_OUT_CH * 1 * KERNEL_SIZE*KERNEL_SIZE);
    loadWeights("biases_layer_0.bin",  d_C1_bias,    C1_OUT_CH);

    // C3
    float *d_C3_kernels, *d_C3_bias;
    cudaMallocManaged(&d_C3_kernels, C3_OUT_CH * C1_OUT_CH * KERNEL_SIZE*KERNEL_SIZE * sizeof(float));
    cudaMallocManaged(&d_C3_bias,    C3_OUT_CH * sizeof(float));
    loadWeights("weights_layer_2.bin", d_C3_kernels, C3_OUT_CH*C1_OUT_CH*KERNEL_SIZE*KERNEL_SIZE);
    loadWeights("biases_layer_2.bin",  d_C3_bias,    C3_OUT_CH);

    // FC1
    float *d_FC1_weights, *d_FC1_bias;
    cudaMallocManaged(&d_FC1_weights, FC1_SIZE * 400 * sizeof(float));
    cudaMallocManaged(&d_FC1_bias,    FC1_SIZE * sizeof(float));
    loadWeights("weights_layer_5.bin", d_FC1_weights, FC1_SIZE*400);
    loadWeights("biases_layer_5.bin",  d_FC1_bias,    FC1_SIZE);

    // FC2
    float *d_FC2_weights, *d_FC2_bias;
    cudaMallocManaged(&d_FC2_weights, FC2_SIZE * FC1_SIZE * sizeof(float));
    cudaMallocManaged(&d_FC2_bias,    FC2_SIZE * sizeof(float));
    loadWeights("weights_layer_6.bin", d_FC2_weights, FC2_SIZE*FC1_SIZE);
    loadWeights("biases_layer_6.bin",  d_FC2_bias,    FC2_SIZE);

    // FC3
    float *d_FC3_weights, *d_FC3_bias;
    cudaMallocManaged(&d_FC3_weights, FC3_SIZE * FC2_SIZE * sizeof(float));
    cudaMallocManaged(&d_FC3_bias,    FC3_SIZE * sizeof(float));
    loadWeights("weights_layer_7.bin", d_FC3_weights, FC3_SIZE*FC2_SIZE);
    loadWeights("biases_layer_7.bin",  d_FC3_bias,    FC3_SIZE);

    /*****************************************************/
    /* 4) C1: convolution (1->6)                         */
    /*****************************************************/
    float* d_C1_output;
    cudaMallocManaged(&d_C1_output, C1_OUT_CH * C1_OUT_SIZE * C1_OUT_SIZE * sizeof(float));
    {
        dim3 threads(16,16,1);
        dim3 blocks(
            (C1_OUT_SIZE+threads.x-1)/threads.x,
            (C1_OUT_SIZE+threads.y-1)/threads.y,
            C1_OUT_CH
        );
        cudaConv2DMultiChannel<<<blocks, threads>>>(
            d_inputImage,
            d_C1_output,
            d_C1_kernels,
            d_C1_bias,
            1,           // in_channels
            RAW_SIZE,    // in_size
            KERNEL_SIZE,
            C1_OUT_CH,
            C1_OUT_SIZE
        );
        cudaDeviceSynchronize();
    }

    /*****************************************************/
    /* 5) S2: pooling (6@28x28 -> 6@14x14) (CPU)         */
    /*****************************************************/
    float* h_S2_output = (float*)malloc(C1_OUT_CH * S1_OUT_SIZE * S1_OUT_SIZE * sizeof(float));
    subsampling2D(d_C1_output, h_S2_output, C1_OUT_SIZE, S1_OUT_SIZE, C1_OUT_CH);

    /*****************************************************/
    /* 6) C3: convolution (6->16) => 16@10x10            */
    /*****************************************************/
    float* d_C3_output;
    cudaMallocManaged(&d_C3_output, C3_OUT_CH * C3_OUT_SIZE * C3_OUT_SIZE * sizeof(float));
    {
        // Il faut copier S2 (CPU) -> GPU
        float* d_S2;
        cudaMallocManaged(&d_S2, C1_OUT_CH * S1_OUT_SIZE * S1_OUT_SIZE * sizeof(float));
        for(int i=0; i<C1_OUT_CH*S1_OUT_SIZE*S1_OUT_SIZE; i++){
            d_S2[i] = h_S2_output[i];
        }

        dim3 threads(16,16,1);
        dim3 blocks(
            (C3_OUT_SIZE+threads.x-1)/threads.x,
            (C3_OUT_SIZE+threads.y-1)/threads.y,
            C3_OUT_CH
        );
        cudaConv2DMultiChannel<<<blocks, threads>>>(
            d_S2,
            d_C3_output,
            d_C3_kernels,
            d_C3_bias,
            C1_OUT_CH,
            S1_OUT_SIZE,
            KERNEL_SIZE,
            C3_OUT_CH,
            C3_OUT_SIZE
        );
        cudaDeviceSynchronize();
        cudaFree(d_S2);
    }

    /*****************************************************/
    /* 7) S4: pooling (16@10x10 -> 16@5x5) (CPU)         */
    /*****************************************************/
    float* h_S4_output = (float*)malloc(C3_OUT_CH * S4_OUT_SIZE * S4_OUT_SIZE * sizeof(float));
    subsampling2D(d_C3_output, h_S4_output, C3_OUT_SIZE, S4_OUT_SIZE, C3_OUT_CH);

    /*****************************************************/
    /* 8) Flatten (16@5x5 -> 400) (CPU->GPU)            */
    /*****************************************************/
    float* h_flatten = (float*)malloc(400*sizeof(float));
    flattenCPU(h_S4_output, h_flatten, C3_OUT_CH, S4_OUT_SIZE, S4_OUT_SIZE);

    float* d_flatten;
    cudaMallocManaged(&d_flatten, 400*sizeof(float));
    for(int i=0; i<400; i++){
        d_flatten[i] = h_flatten[i];
    }

    /*****************************************************/
    /* 9) FC1: 400 -> 120 + tanh (GPU)                  */
    /*****************************************************/
    float* d_FC1_output;
    cudaMallocManaged(&d_FC1_output, FC1_SIZE*sizeof(float));
    {
        int blockSize = 128;
        int gridSize  = (FC1_SIZE + blockSize - 1)/blockSize;
        fullyConnected120<<<gridSize, blockSize>>>(
            d_flatten, d_FC1_weights, d_FC1_bias,
            d_FC1_output, 400
        );
        cudaDeviceSynchronize();
    }

    /*****************************************************/
    /* 10) FC2: 120 -> 84 + tanh (GPU)                  */
    /*****************************************************/
    float* d_FC2_output;
    cudaMallocManaged(&d_FC2_output, FC2_SIZE*sizeof(float));
    {
        int blockSize = 128;
        int gridSize  = (FC2_SIZE + blockSize - 1)/blockSize;
        fullyConnected84<<<gridSize, blockSize>>>(
            d_FC1_output, d_FC2_weights, d_FC2_bias,
            d_FC2_output, FC1_SIZE
        );
        cudaDeviceSynchronize();
    }

    /*****************************************************/
    /* 11) FC3: 84 -> 10  => softmax (GPU)              */
    /*****************************************************/
    float* d_FC3_output;
    cudaMallocManaged(&d_FC3_output, FC3_SIZE*sizeof(float));
    {
        int blockSize = 128;
        int gridSize  = (FC3_SIZE + blockSize - 1)/blockSize;
        fullyConnected10<<<gridSize, blockSize>>>(
            d_FC2_output, d_FC3_weights, d_FC3_bias,
            d_FC3_output, FC2_SIZE
        );
        cudaDeviceSynchronize();
    }

    float* d_softmax;
    cudaMallocManaged(&d_softmax, FC3_SIZE*sizeof(float));
    {
        int blockSize = 32; // 10 < 32
        int gridSize  = 1;
        softmaxKernel<<<gridSize, blockSize>>>(d_FC3_output, d_softmax, FC3_SIZE);
        cudaDeviceSynchronize();
    }

    /*****************************************************/
    /* 12) Argmax => prédiction                          */
    /*****************************************************/
    int predicted = argmaxCPU(d_softmax, FC3_SIZE);
    printf("\n=== PREDICTION = %d ===\n\n", predicted);

    /*****************************************************/
    /* 13) Libérations mémoire                           */
    /*****************************************************/
    // CPU
    free(h_S2_output);
    free(h_S4_output);
    free(h_flatten);

    // GPU
    cudaFree(d_inputImage);
    cudaFree(d_C1_kernels);
    cudaFree(d_C1_bias);
    cudaFree(d_C1_output);
    cudaFree(d_C3_kernels);
    cudaFree(d_C3_bias);
    cudaFree(d_C3_output);
    cudaFree(d_flatten);
    cudaFree(d_FC1_weights);
    cudaFree(d_FC1_bias);
    cudaFree(d_FC1_output);
    cudaFree(d_FC2_weights);
    cudaFree(d_FC2_bias);
    cudaFree(d_FC2_output);
    cudaFree(d_FC3_weights);
    cudaFree(d_FC3_bias);
    cudaFree(d_FC3_output);
    cudaFree(d_softmax);

    return 0;
}

/****************************************************/
/*           Lecture MNIST + endianness            */
/****************************************************/
unsigned int swapEndian(unsigned int value) {
    return ((value >> 24) & 0xff)
         | ((value << 8) & 0xff0000)
         | ((value >> 8) & 0xff00)
         | ((value << 24) & 0xff000000);
}

float* readMNISTImage(const char* filename, int imgIndex) {
    FILE* fptr = fopen(filename, "rb");
    if (!fptr) {
        printf("Impossible d'ouvrir le fichier : %s\n", filename);
        exit(1);
    }

    unsigned int magic, nbImg, nbRows, nbCols;
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    magic = swapEndian(magic);
    nbImg = swapEndian(nbImg);
    nbRows = swapEndian(nbRows);
    nbCols = swapEndian(nbCols);

    if (magic != 2051) {
        printf("Erreur: fichier MNIST invalide (magic=%u)\n", magic);
        fclose(fptr);
        exit(1);
    }
    if (nbRows != 28 || nbCols != 28) {
        printf("Erreur: dimensions attendues 28x28\n");
        fclose(fptr);
        exit(1);
    }
    if (imgIndex < 0 || imgIndex >= (int)nbImg) {
        printf("Index d'image invalide (%d), max=%u\n", imgIndex, nbImg-1);
        fclose(fptr);
        exit(1);
    }

    // Sauter imgIndex images
    fseek(fptr, imgIndex * 28 * 28, SEEK_CUR);

    float* image = (float*)malloc(28 * 28 * sizeof(float));
    for(int i=0; i<28*28; i++){
        unsigned char val;
        fread(&val, 1, 1, fptr);
        // Normalisation [0..1]
        image[i] = (float)val / 255.0f;
    }

    fclose(fptr);
    return image;
}

/****************************************************/
/*           Fonctions d'affichage console         */
/****************************************************/
void charBckgrndPrint(char *str, int rgb[3]) {
    // Séquence ANSI pour couleur de fond
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    // Imprimer le "str"
    printf("%s\033[0m", str); // reset
}

void imgColorPrint(int height, int width, int ***img) {
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            charBckgrndPrint("  ", img[i][j]); 
        }
        printf("\n");
    }
}

int ***allocImgRGB(int height, int width){
    int ***img = (int ***)malloc(height * sizeof(int**));
    for(int i=0; i<height; i++){
        img[i] = (int **)malloc(width*sizeof(int*));
        for(int j=0; j<width; j++){
            img[i][j] = (int*)malloc(3*sizeof(int));
        }
    }
    return img;
}
void freeImgRGB(int ***img, int height, int width){
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);
}

/****************************************************/
/*      Chargement des poids / biais .bin          */
/****************************************************/
void loadWeights(const char *file_path, float *host_data, int size) {
    FILE *file = fopen(file_path, "rb");
    if(file == NULL){
        printf("Erreur: impossible d'ouvrir %s\n", file_path);
        exit(1);
    }
    fread(host_data, sizeof(float), size, file);
    fclose(file);
}

/****************************************************/
/*       Convolution multi-canal (GPU)             */
/****************************************************/
__device__ float activation_tanh(float x){
    return tanhf(x);
}

__global__ void cudaConv2DMultiChannel(
    float *input,    
    float *output,   
    float *kernels,  
    float *bias,     
    int in_channels,
    int input_size,
    int kernel_size,
    int out_channels,
    int output_size
){
    int oc = blockIdx.z;
    int out_i = blockIdx.y * blockDim.y + threadIdx.y;
    int out_j = blockIdx.x * blockDim.x + threadIdx.x;

    if(oc < out_channels && out_i < output_size && out_j < output_size){
        float sum = bias[oc];
        for(int ic=0; ic<in_channels; ic++){
            for(int ki=0; ki<kernel_size; ki++){
                for(int kj=0; kj<kernel_size; kj++){
                    int in_i = out_i + ki;
                    int in_j = out_j + kj;
                    float val = input[ic*(input_size*input_size) + in_i*input_size + in_j];
                    float w = kernels[
                        oc*(in_channels*kernel_size*kernel_size)
                        + ic*(kernel_size*kernel_size)
                        + ki*kernel_size
                        + kj
                    ];
                    sum += val * w;
                }
            }
        }
        // activation
        output[oc*(output_size*output_size) + out_i*output_size + out_j] = activation_tanh(sum);
    }
}

/****************************************************/
/*     Pooling 2x2, CPU                             */
/****************************************************/
void subsampling2D(float *input, float *output, 
                   int input_size, int output_size,
                   int num_channels)
{
    for (int c = 0; c < num_channels; c++){
        for(int i=0; i<output_size; i++){
            for(int j=0; j<output_size; j++){
                int in_i = i*2;
                int in_j = j*2;
                float sum=0.0f;
                for(int ki=0; ki<2; ki++){
                    for(int kj=0; kj<2; kj++){
                        sum += input[c*input_size*input_size + (in_i+ki)*input_size + (in_j+kj)];
                    }
                }
                output[c*output_size*output_size + i*output_size + j] = sum / 4.0f;
            }
        }
    }
}

/****************************************************/
/*     Flatten 16@5x5 => 400 (CPU)                  */
/****************************************************/
void flattenCPU(float* input, float* output, 
                int channels, int height, int width)
{
    int idx=0;
    for(int c=0; c<channels; c++){
        for(int i=0; i<height; i++){
            for(int j=0; j<width; j++){
                output[idx++] = input[c*height*width + i*width + j];
            }
        }
    }
}

/****************************************************/
/*      FC layers (GPU)                             */
/****************************************************/
__global__ void fullyConnected120(float* input, float* weights, float* biases,
                                  float* output, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < 120){
        float sum = biases[idx];
        for(int i=0; i<input_size; i++){
            sum += input[i] * weights[idx*input_size + i];
        }
        output[idx] = tanhf(sum);
    }
}

__global__ void fullyConnected84(float* input, float* weights, float* biases,
                                 float* output, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < 84){
        float sum = biases[idx];
        for(int i=0; i<input_size; i++){
            sum += input[i] * weights[idx*input_size + i];
        }
        output[idx] = tanhf(sum);
    }
}

__global__ void fullyConnected10(float* input, float* weights, float* biases,
                                 float* output, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < 10){
        float sum = biases[idx];
        for(int i=0; i<input_size; i++){
            sum += input[i] * weights[idx*input_size + i];
        }
        // pas de tanh, on applique softmax après
        output[idx] = sum;
    }
}

/****************************************************/
/*    Softmax (GPU)                                 */
/****************************************************/
__global__ void softmaxKernel(float* input, float* output, int size){
    __shared__ float tmp[32]; // 10 < 32
    int tid = threadIdx.x;

    if(tid < size){
        tmp[tid] = expf(input[tid]);
    }
    __syncthreads();

    float sum=0.0f;
    if(tid==0){
        for(int i=0; i<size; i++){
            sum += tmp[i];
        }
        tmp[0] = sum;  // on stocke la somme en tmp[0]
    }
    __syncthreads();

    if(tid < size){
        output[tid] = tmp[tid] / tmp[0];
    }
}

/****************************************************/
/*     argmax CPU                                   */
/****************************************************/
int argmaxCPU(float* data, int size){
    int bestIdx=0;
    float bestVal = data[0];
    for(int i=1; i<size; i++){
        if(data[i] > bestVal){
            bestVal = data[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}
