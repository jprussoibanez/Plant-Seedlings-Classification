# 1. Kagglers solutions and discussions
The first step is to review the discussion forum for this competition to better discover the challenges and approaches in solving the problem.

## Discussion forum
The following are the interesting discussions threads:
1. [Can Top People in the LeaderBoard share their apporoach?](https://www.kaggle.com/c/plant-seedlings-classification/discussion/53021). This links to [Kumar Sridhar Medium Post](https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86) explaining how he reach the 5th position on the leaderboard.
2. [Plant Segmentation does not effect (that much) accuracy](https://www.kaggle.com/c/plant-seedlings-classification/discussion/45271). This discussion shows finally that segmentation do have an increase on acurracy [3-8]% and links to a [kernel](https://www.kaggle.com/gaborvecsei/plant-seedlings-fun-with-computer-vision) with the solution.
3. [Some Tips to Improve accuracy](https://www.kaggle.com/c/plant-seedlings-classification/discussion/46699). This a discussion on image size and pre-trained models to improve accuracy. The recommendation is to use image sizes between 300-400pxs and simple pre-trained models like ResNet or Inception. [Links](https://github.com/GodsDusk/Plant-Seedlings-Classification) to another solution from Kaggler positioned 47th on the leaderboard. Also explains how to use [pre-trained models with different input image sizes](https://stackoverflow.com/questions/44161967/keras-vggnet-pretrained-model-variable-sized-input). Another [discussion](https://www.kaggle.com/c/plant-seedlings-classification/discussion/45206) confirms better accuracy between 200-400pxs on image size.
4. [image background generalization](https://www.kaggle.com/c/plant-seedlings-classification/discussion/50323). This links to a kernel for [image segmentation and background removal](https://www.kaggle.com/ianchute/background-removal-cieluv-color-thresholding)
5. [Test images from the wild](https://www.kaggle.com/c/plant-seedlings-classification/discussion/44490). Author from the competition dataset shares the link to the [dataset webpage](https://vision.eng.au.dk/plant-seedlings-dataset/).

## Kagglers Kernels
The following are interesting sample kagglers kernels.
1. [Keras simple model (0.97103 Best Public Score)](https://www.kaggle.com/miklgr500/keras-simple-model-0-97103-best-public-score) This shows a simple custom deep network can achieve high scores.
2. [Seedlings - Pretrained keras models](https://www.kaggle.com/gaborfodor/seedlings-pretrained-keras-models). Kernel showing transfer learning with a Xception pre-trained model (around 83.6% accuracy).
3. [Plants PCA & t-SNE ( w/ image scatter plot)](https://www.kaggle.com/gaborvecsei/plants-t-sne) Nice data visualization with t-SNE.
4. [Plants Xception 90.06% Test Accuracy](https://www.kaggle.com/raoulma/plants-xception-90-06-test-accuracy). Kernel with transfer learning on Xception and different classification models with Logistic Regression, random forest and fully connect neural network.
5. [CNN + SVM + XGBoost](https://www.kaggle.com/matrixb/cnn-svm-xgboost) Combine other classifiers (SVM, XGBoost, Logistic regression, etc.) with the convolutional layers

## Conclusions
From the discussion forums and kernels we can conclude that:
1. There is evidence of high scores with simple pre-trained models like Xception or ResNet  and also on custom multi-layer CNN with FNN classifier. The recommendation is to avoid more complex models like DenseNet or ResNext. 
2. Kagglers add different classifiers like XGBoost, SVM, Logistic Regression and Random Forest with different results. Those classifiers build on top of the CNN in combination with the FNN classifier.
3. Segmentation has shown to improve performance. As images are specially photograph for training purposes it seems there is no need to crop by applying [bounding boxes](https://www.kaggle.com/martinpiotte/bounding-box-data-for-the-whale-flukes) like in this [whale competition](https://www.kaggle.com/c/humpback-whale-identification).
4. The seedling dataset have a small ammount of images with imbalanced classes, so data augmentation and balancing techniques should be used.

All this assumptions will be validated throughout the data analysis.