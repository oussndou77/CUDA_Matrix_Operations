#include <stdio.h>
#include <stdlib.h>

#define WIDTH 28
#define HEIGHT 28

// Fonction pour afficher un pixel avec une couleur de fond en console
void charBckgrndPrint(char *str, int rgb[3]) {
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m", str);
}

// Fonction pour afficher une image en couleurs dans la console
void imgColorPrint(int height, int width, int ***img) {
    int row, col;
    char *str = "  ";
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            charBckgrndPrint(str, img[row][col]);
        }
        printf("\n");
    }
}

unsigned int swapEndian(unsigned int value) {
    return ((value >> 24) & 0xff) |      // Byte 3 to Byte 0
           ((value << 8) & 0xff0000) |  // Byte 1 to Byte 2
           ((value >> 8) & 0xff00) |    // Byte 2 to Byte 1
           ((value << 24) & 0xff000000); // Byte 0 to Byte 3
}


// Fonction pour lire les données MNIST et afficher une image
void readAndDisplayMNIST(const char *filename, int imgIndex) {
    int i, j;
    int ***img;
    int color[3] = {255, 255, 255}; // Blanc pour les pixels
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

    // Allocation mémoire pour l'image
    img = (int ***)malloc(HEIGHT * sizeof(int **));
    for (i = 0; i < HEIGHT; i++) {
        img[i] = (int **)malloc(WIDTH * sizeof(int *));
        for (j = 0; j < WIDTH; j++) {
            img[i][j] = (int *)malloc(sizeof(int) * 3);
        }
    }

    // Ouvrir le fichier
    if ((fptr = fopen(filename, "rb")) == NULL) {
        printf("Impossible d'ouvrir le fichier : %s\n", filename);
        exit(1);
    }

    // Lecture de l'entête MNIST
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    

    magic = swapEndian(magic);
    nbImg = swapEndian(nbImg);
    nbRows = swapEndian(nbRows);
    nbCols = swapEndian(nbCols);


    printf("Magic Number: %u\n", magic);
    printf("Nombre d'Images: %u\n", nbImg);
    printf("Dimensions: %ux%u\n", nbRows, nbCols);

    // Passer aux données de l'image demandée
    fseek(fptr, imgIndex * HEIGHT * WIDTH, SEEK_CUR);

    // Lecture des données de l'image
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr);
            int intensity = (int)((255 - val) * color[0] / 255); // Gris inversé
            img[i][j][0] = intensity;
            img[i][j][1] = intensity;
            img[i][j][2] = intensity;
        }
    }

    // Afficher l'image
    imgColorPrint(HEIGHT, WIDTH, img);

    // Libérer la mémoire
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);

    fclose(fptr);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage : %s <fichier MNIST> <index image>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int imgIndex = atoi(argv[2]);

    readAndDisplayMNIST(filename, imgIndex);

    return 0;
}
