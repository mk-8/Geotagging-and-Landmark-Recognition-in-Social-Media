# Geotagging and Landmark Recognition in Social Media
In this project we solve a `multi-label-classification` problem by classifying/tagging a given image of a famous landmark using CNN (Convolutional Neural Network).

## Features
⚡Multi Label Image Classification  
⚡Custom CNN  
⚡Transfer Learning CNN  
⚡PyTorch

## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)


## Introduction


Photo sharing and photo storage services like to have location data for each uploaded photo. In addition, these services can build advanced features with the location data, such as the automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. However, although a photo's location can often be obtained by looking at the photo's metadata, many images uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the picture. However, given the large number of landmarks worldwide and the immense volume of images uploaded to photo-sharing services, using human judgment to classify these landmarks would not be feasible. In this project, we'll try to address this problem by building `Neural Network` (NN) based models to automatically predict the location of the image based on any landmarks depicted in the picture.

## Objective
To build NN based model that'd accept any user-supplied image as input and suggest the `top k` most relevant landmarks from '50 possible` landmarks from across the world. 

1. Download the dataset 
2. Build a CNN based neural network from scratch to classify the landmark image
   - Here, we aim to attain a test accuracy of at least 30%. At first glance, an accuracy of 30% may appear to be very low, but it's way better than random guessing, which would provide an accuracy of just 2% since we have 50 different landmarks classes in the dataset.
3. Build a CNN based neural network, using transfer-learning, to classify the landmark image
    - Here, we aim to attain a test accuracy of at least 60%, which is pretty good given the complex nature of this task.
4. Implement an inference function that will accept a file path to an image and an integer k and then predict the top k most likely landmarks this image belongs to. The print below displays the expected sample output from the predict function, indicating the top 3 (k = 3) possibilities for the image in question.

<img src="./assets/sample_output.png">

## Dataset
- Dataset to be downloaded from [here](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip). Note that this is a mini dataset containing around 6,000 images); this dataset is a small subset of the [Original Landmark Dataset](https://github.com/cvdfoundation/google-landmark) that has over 700,000 images.
- Unzipped dataset would have the parent folder `landmark_images` containing training data in the `train` sub-folder and testing data in the `test` sub-folder
- There are 1250 images in the `test` sub-folder to be kept hidden and only used for model evaluation
- There are 4996 images in the `train` sub-folder to be used for training and validation
- Images in `test` and `train` sets are further categorized and kept in one of the 50 sub-folders representing 50 different landmarks classes (from 0 to 49)
- Images in the dataset are of different sizes and resolution
- Here are a few samples from the training dataset with their respective labels descriptions...

![Landmark Samples](assets/landmark_samples.png)

## Evaluation Criteria

### Loss Function  
We will use `LogSoftmax` in the output layer of the network...

<img src="assets/LogSoftmax.png">

We need a suitable loss function that consumes these `log-probabilities` outputs and produces a total loss. The function that we are looking for is `NLLLoss` (Negative Log-Likelihood Loss). In practice, `NLLLoss` is nothing but a generalization of `BCELoss` (Binary Cross EntropyLoss or Log Loss) extended from binary-class to multi-class problem.

<img src="assets/NLLLoss.png">

<br>Note the `negative` sign in front `NLLLoss` formula hence negative in the name. The negative sign is put in front to make the average loss positive. Suppose we don't do this then since the `log` of a number less than 1 is negative. In that case, we will have a negative overall average loss. To reduce the loss, we need to `maximize` the loss function instead of `minimizing,` which is a much easier task mathematically than `maximizing.`


### Performance Metric

`accuracy` is used as the model's performance metric on the test-set 

<img src="assets/accuracy.png">


## Solution Approach
- Once the dataset is downloaded and unzipped, we split the training set into training and validation sets in 80%:20% (3996:1000) ratio and keep images in respective `train` and `val` sub-folders.
- `train` data is then used to build Pytorch `Dataset` object; after applying data augmentations, images are resized to 128x128.
`mean` and `standard deviation` is computed for the train dataset, and then the dataset is `normalized` using the calculated statistics. 
- The RGB channel histogram of the train set is shown below...

<img src="assets/train_hist1.png">

- The RGB channel histogram of the train set after normalization is shown below...

<img src="assets/train_hist2.png">

- Now, `test` and `val` Dataset objects are prepared in the same fashion where images are resized to 128x128 and then normalized.
- The training, validation, and testing datasets are then wrapped in Pytorch `DataLoader` object so that we can iterate through them with ease. A typical `batch_size` 32 is used.

