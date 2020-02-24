# The project

The problem we want to analyze is the [Kaggle plant seedlings classification](https://www.kaggle.com/c/plant-seedlings-classification).
##  Competition goal
The goal is to differentiate a weed from a crop seedling in order to perform site-specific weed control.

## Dataset
The [database of images](https://arxiv.org/abs/1711.05458) has approximately 960 unique plants belonging to 12 species at several growth stages. It comprises annotated RGB images with a physical resolution of roughly 10 pixels per mm.

## Kernel structure
The following is a summary of the kernel main structure.

### [1. Kagglers solutions and discussions](docs/kagglers_discussions.md)
- Review other kagglers kernels to look into possible solutions.
- Review the competition discussion forum for better understanding of the challenges and general ideas.

Further discussion and conclusions can be find [here](docs/kagglers_discussions.md).

### [2. Libraries and settings](docs/settings.md)
- This section has all available settings to configure the training and model configuration.

You can find a better explanation for each configuration [here](docs/settings.md).
### 3. Data analysis◊
- Main data exploration and analysis to determine the best model to solve the challenge.
- Use of descriptive analysis to determine dataset distribution.
- Use of t-SNE to reduce dimensionality for data visualization.
### 4. Pre-processing
- Use class weights to balance the dataset.
- Image segmentation to remove image soil background.
- Data augmentation to increase the dataset.
### 5. Processing

- Use of transfer learning with different pre-trained networks Resnet50 and InceptionV3. Other networks can be easily added.
- Use of custom simple CNN. This configuration can be improve by using hyperparameters optimization like [hyperopt](https://github.com/hyperopt/hyperopt).
- Use of a simple FNN as a classifier. This can be improved by using other classifiers like SVM, XGBoost, etc.

### 6. Generate prediction file

- Generate prediction file with Kaggle competition format.

# Links

This are some papers and links use during solving the exercise:

- [Deep Learning using Linear Support Vector Machines](https://arxiv.org/pdf/1306.0239.pdf)
- [A New Design Based-SVM of the CNN Classifier Architecture with Dropout for Offline Arabic Handwritten Recognition](https://www.sciencedirect.com/science/article/pii/S1877050916309991)
- [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
- [Regularization and Optimization strategies in Deep Convolutional Neural Network](https://arxiv.org/pdf/1712.04711.pdf)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [Revisiting small batch training for deep neural networks](https://arxiv.org/pdf/1804.07612.pdf)