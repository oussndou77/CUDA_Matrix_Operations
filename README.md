# 🚀 CUDA Matrix Operations & LeNet-5 Implementation

## 📖 Description
Ce projet explore l'utilisation de **CUDA** pour effectuer des opérations matricielles parallèles sur GPU et l'implémentation du réseau de neurones convolutif **LeNet-5**. L'objectif est de comprendre l'accélération offerte par les GPU pour des calculs intensifs et de simuler un réseau convolutif complet.

---

## 🎯 Objectifs
- 💡 Comprendre le parallélisme avec CUDA.
- ⚡ Comparer les performances entre **CPU** et **GPU**.
- 🧠 Implémenter un réseau convolutif (**LeNet-5**) en C/CUDA.
- 🔍 Étudier les résultats et performances sur le dataset **MNIST**.

---

## ✨ Fonctionnalités
### Partie 1 : Opérations sur Matrices
1. Initialisation de matrices avec des valeurs aléatoires.
2. Affichage des matrices sur CPU.
3. Addition et multiplication de matrices :
   - Sur **CPU**.
   - Sur **GPU** avec CUDA.

### Partie 2 : Implémentation du LeNet-5
1. Génération des données d'entrée pour MNIST.
2. Convolution 2D avec fonction d'activation **tanh**.
3. Sous-échantillonnage (moyennage 2x2).
4. Couches fully connected et activation **softmax**.
5. Tests avec les données MNIST et comparaison avec Python.

---

## 🛠️ Prérequis
- **CUDA Toolkit** (NVCC Compiler).
- Une carte graphique NVIDIA compatible CUDA.
- **MNIST Dataset** (fichiers binaires : `train-images.idx3-ubyte` et `train-labels.idx1-ubyte`).
- Environnement Linux ou Windows avec un éditeur compatible.

---

## 🏃 Instructions pour les Tests
### Étape 1 : Compiler le programme
Utilisez la commande suivante pour compiler le fichier principal :
```bash
nvcc main.cu -o main
```

### Étape 2 : Lancer les tests
Pour effectuer les tests, exécutez la commande suivante avec les chemins appropriés pour les fichiers MNIST :
```bash
./main train-images.idx3-ubyte train-labels.idx1-ubyte
```

### Étape 3 : Tester une image spécifique
Pour tester une image particulière, ajoutez l'indice de l'image :
```bash
./main train-images.idx3-ubyte train-labels.idx1-ubyte 0
```

# 📊 Résultats des Tests

## Partie 1 : Opérations sur Matrices

| Taille \(n\) | Addition CPU (ms) | Addition GPU (ms) | Multiplication CPU (ms) | Multiplication GPU (ms) |
|--------------|--------------------|--------------------|--------------------------|--------------------------|
| 10           | 0                  | 0.33               | 2                        | 0.029                    |
| 1000         | 2                  | 0.296              | 2380                     | 2.802                    |
| 10000        | 216                | 7.445              | Pas de valeur (n trop grand) | 2789                 |

## Partie 2 : Implémentation du LeNet-5

- **Prédiction MNIST** : Le réseau prédit toujours la classe 0.

### Analyse des causes :

- Problèmes potentiels dans le chargement des poids ou le prétraitement des images.
- Différences dans l'architecture entre Python et CUDA.

### Comparaison des performances GPU vs CPU :

- Accélération significative obtenue sur GPU pour les étapes convolutives et fully connected.

---

# 🌟 Ce que nous avons appris

- L'importance de l'alignement des poids, biais et dimensions entre le modèle Python et CUDA.
- L'efficacité du GPU pour les calculs massivement parallèles, notamment dans l'apprentissage profond.
- Les défis de l'implémentation manuelle d'un réseau convolutif, tels que le prétraitement des données et la gestion des paramètres.

---

# 💬 Remerciements

Ce projet a été réalisé avec mon binôme de TP, Mathys BARRIE, dont les idées et la collaboration ont été essentielles à la réussite de ce travail. 🙌

# 📁 Plus de détails

Si vous souhaitez en savoir plus sur nos résultats, n'hésitez pas à consulter le dossier **Compte-Rendu** 📚, où vous trouverez un résumé détaillé de notre travail. 🔍
