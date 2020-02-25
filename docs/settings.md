# 2. Libraries and settings

## 2.1 Libraries

This are the main libraries for coding:

1. [tensorflow](https://www.tensorflow.org/) with [keras](https://keras.io/) for managing the deep learning models.
2. [sckilit-learn](https://scikit-learn.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/) with [seaborn](https://seaborn.pydata.org/) and [matplotlib](https://matplotlib.org/) for data manipulation, analysis and visualization.
3. [TQDM](https://github.com/tqdm/tqdm) for progress bar visualization on processing.
4. [opencv](https://pypi.org/project/opencv-python/) for image processing.

## 2.2 Settings

The following are all the main configuration settings to train and run the model.

## 2.2.1 Global Settings

This are the global variables use to setup the configuration for the models.

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Example (default)</th>
</tr>
</thead>
<tbody>
<tr>
<td>BASE_DATASET_FOLDER</td>
<td>Base path to dataset folder</td>
<td>../input/plant-seedlings-classification</td>
</tr>
<tr>
<td>TRAIN_DATASET_FOLDER</td>
<td>Train dataset folder inside base path</td>
<td>train</td>
</tr>
<tr>
<td>TEST_DATASET_FOLDER</td>
<td>Test dataset folder inside base path</td>
<td>test</td>
</tr>
<tr>
<td>IMAGE_WIDTH</td>
<td>Image width in pixels for training on CNN. Images will be resize to image_width and image_height</td>
<td>299</td>
</tr>
<tr>
<td>IMAGE_HEIGHT</td>
<td>Image heigh in pixels for training on CNN. Images will be resize to image_width and image_height</td>
<td>299</td>
</tr>
<tr>
<td>IMAGE_CHANNELS</td>
<td>Image channels input for the CNN</td>
<td>3</td>
</tr>
<tr>
<td>TSNE_VISUALIZATION</td>
<td>t-SNE visualization can take some time to perform so this is a toggle for using it</td>
<td>True</td>
</tr>
</tbody>
</table>

## 2.2.2 Data agumentation settings

This are the settings use to setup the data augmentation parameters.

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Example (default)</th>
</tr>
</thead>
<tbody>
<tr>
<td>rotation_range</td>
<td>Degree range for random rotations.</td>
<td>250</td>
</tr>
<tr>
<td>width_shift_range</td>
<td>Percentage of range for width shifting</td>
<td>0.5</td>
</tr>
<tr>
<td>height_shift_range</td>
<td>Percentage of range for width shifting</td>
<td>0.5</td>
</tr>
<tr>
<td>horizontal_flip</td>
<td>Randomly flip inputs horizontally</td>
<td>True</td>
</tr>
<tr>
<td>vertical_flip</td>
<td>Randomly flip inputs vertically</td>
<td>True</td>
</tr>
</tbody>
</table>

## 2.2.3 Training settings

This are the settings use to setup the data augmentation parameters.

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Example (default)</th>
</tr>
</thead>
<tbody>
<tr>
<td>batch_size</td>
<td>Size for batching the training process</td>
<td>50</td>
</tr>
<tr>
<td>epochs</td>
<td>Number of epochs to ran the training process. The training will be using early stopping</td>
<td>500</td>
</tr>
<tr>
<td>steps_per_epoch</td>
<td>Steps for each epoch. This can be calculated by the ammount of data to process and the batch size</td>
<td>150</td>
</tr>
<tr>
<td>patience</td>
<td>Patience for early stopping</td>
<td>7</td>
</tr>
</tbody>
</table>

## 2.2.4 Network settings

This are the settings to setup the network to train and use.

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Example (default)</th>
</tr>
</thead>
<tbody>
<tr>
<td>NETWORK_TO_TRAIN</td>
<td>
    The are three networks to train and use:</br>
<ul>
    <li>CUSTOM_CNN: Simple three layer CNN</li>
    <li>RESNET_50: <a hred="https://keras.io/applications/#resnet">Resnet50 from keras applications</a></li>
    <li>INCEPTION_V3: <a hred="https://keras.io/applications/#xception">Inception from keras applications</a></li>
</ul>
</td>
<td>CUSTOM_CNN</td>
</tr>
</tbody>
</table>