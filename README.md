# The project
The problem to analyze is the [Kaggle plant seedlings classification](https://www.kaggle.com/c/plant-seedlings-classification).

##  Competition goal
The goal is to differentiate a weed from a crop seedling in order to perform site-specific weed control.

## Dataset
The [database of images](https://arxiv.org/abs/1711.05458) has approximately 960 unique plants belonging to 12 species at several growth stages. It comprises annotated RGB images with a physical resolution of roughly 10 pixels per mm.

## Kernel structure
The following is a summary of the kernel main structure.

### 1. Kagglers challenges and discussions
- Review other kagglers kernels to better understand the challenges.
- Review the competition discussion forum for interesting threads.
- Generate some insights and conclusions from the ideas use in solving the problem.

### 2. Libraries and settings
- This section has all available settings to configure the model and its training parameters.

### 3. Data analysis
- Main data exploration and analysis to determine the best model to solve the problem.
- Use of descriptive analysis to determine dataset distribution.
- Use of t-SNE to reduce dimensionality for data visualization.

### 4. Pre-processing
- Class weights definition to balance the dataset distribution.
- Image segmentation to remove image soil background.
- Data augmentation to increase the images dataset.

### 5. Processing
- Use of transfer learning with different pre-trained models like Resnet50 and InceptionV3. Other pre-trained models can be easily added.
- Use of custom CNN with multiple layers.
- FNN as the last layer classifier.

### 6. Generate prediction file
- Generate prediction file with Kaggle competition format.

# ToDo & Improvements
- Fine tune pre-trained models by unfreezing and training the last layers on the CNN.
- Use hyperparameters optimization using some library like [hyperopt](https://github.com/hyperopt/hyperopt) to optimize model parameters.
- Use more classifiers like XGBoost, SVM, etc. instead of just the FNN.
- Ensemble models to improve performance by combining different models.
- Use cross-validation to better evaluate the estimator performance.

# Links

This are some papers and links use during the exercise resolution:

- [Deep Learning using Linear Support Vector Machines](https://arxiv.org/pdf/1306.0239.pdf)
- [A New Design Based-SVM of the CNN Classifier Architecture with Dropout for Offline Arabic Handwritten Recognition](https://www.sciencedirect.com/science/article/pii/S1877050916309991)
- [Transfer learning from pre-trained models](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
- [Regularization and Optimization strategies in Deep Convolutional Neural Network](https://arxiv.org/pdf/1712.04711.pdf)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [Revisiting small batch training for deep neural networks](https://arxiv.org/pdf/1804.07612.pdf)