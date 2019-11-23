## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/OriginalImage.jpg "Original Image"
[image3]: ./examples/NormalizedImage.jpg "Normalized Image"
[image4]: ./signs_images/12.jpg "Traffic Sign 1"
[image5]: ./signs_images/6.jpg "Traffic Sign 2"
[image6]: ./signs_images/8.jpg "Traffic Sign 3"
[image7]: ./signs_images/9.jpg "Traffic Sign 4"
[image8]: ./signs_images/5.jpg "Traffic Sign 5"
[image9]: ./examples/Softmax_Visualization.JPG "Softmax Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As described in the class, preprocessing of data is very important to extract the right kind of features from the image, So few techniques worked for my model, and few didn't.

**What I tried** - 

* I converted images to grayscale.
* Tried reducing the size of the image from (32,32) to (26,26) 
* Normalising the images. 

**What I fixed for the model**

* Normalising the images

Here for my model I only used Normalization as a preprocessing technique. I got validation accuracy same for both i.e.; what I tried and what I fixed, but the test accuracy was way too low when I converted the images to grayscale. So I decided only to normalize the images.

**I did not used data augmentation as I was able to get good validation accuracy and testing accuracy.**

![alt text][image2]
![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x6 	|
| RELU		            |                                               |
| Convolution 7x7		| 1x1 stride, valid padding, outputs 20x20x16   |										
| Max pooling	      	| 2x2 stride,  outputs 10x10x16                 |
| RELU                  |                                               |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 6x6x400    |
| Max pooling      		| 2x2 stride, outputs 3x3x400        			|
| Flatten				| Conv1, Conv2, Conv3 , outputs 9256    		|
| Fully Connected		| outputs(43)									|
|						|												|

**CODE SNIPPET**

```python
    conv1_W = tf.Variable(tf.truncated_normal(shape=(7, 7, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    print("Layer 1|Convolutional Layer 7x7x3 : ",conv1.shape)
    
    conv2_W = tf.Variable(tf.truncated_normal(shape=(7, 7, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    print("Convolutional Layer 7x7x6 : ",conv2.shape)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print("Layer 2|Maxpooling Layer 20x20x16 : ",conv2.shape)
    
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(400))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    print("Convolutional Layer 10x10x400 : ", conv3.shape)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print("Layer 3|Maxpooling Layer 6x6x400 : ",conv3.shape)

    flatten_1 = flatten(conv1)
    print("Flatten Layer 1 : ",flatten_1.shape)
    flatten_2 = flatten(conv2)
    print("Flatten Layer 2 : ",flatten_2.shape)
    flatten_3 = flatten(conv3)
    print("Flatten Layer 3 : ",flatten_3.shape)
    x = tf.concat([flatten_1, flatten_2, flatten_3], 1)
    print("Concatenate Flatten layers : ",x.shape)
    
    fc1_W = tf.Variable(tf.truncated_normal(shape=(9256, 43), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(x, fc1_W) + fc1_b
    print("Fully Connected Layer : ",logits.shape)

    return logits
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with initialising the placeholders, `x` stores the input patches and `y` stores the label which is then one hot encoded. 

**Code Snippet**
```python
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
```

Then for the second step I calculated the `cross entropy`. Cross Entropy is a measure of how different the logits are from the groud truth training labels.

Then Averaged the cross entropy and applied Adam Optimizer to minimize the loss function. Finally run the minimize function on the optimizer whixh uses Backpropagation to update the weights and minimize the loss.

**Code Snippet**
```python
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
Below are the Optimizer and Hyper Parameters I used for my model: 

* Type of Optimizer: Adam Optimizer
* Batch Size: 200
* Number of Epochs: 40
* Learning Rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 1.0
* Validation set accuracy of 0.97 
* Test set accuracy of 0.95

As in the classes it was told to preprocess the images which includes converting image to grayscale and normalize the images. I first tried that but I was getting the test accuracy upto 0.62. Then I only normalized images and reached 0.80 accuracy.

I also played with the filter size and number of outputs. With the experience I belive if the filter size is more than we get better feature map also when we go deeper i.e.; more hidden layers it gives better accuracy. 

So here my filter size for layer is 7x7 and 5x5. And input shape passed to the fully connected is 9256. I acheived test accuracy upto 0.90 and validation accuracy upto 0.97 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there's backgroud noise the light behind the sign is illuminating. In the second image there's no light hence can be difficult to classify. In 3rd anf 4th image the images are tilted and is not in the normal angle therfore difficult for model to identify also I have not rotated and scaled images as a preprocessing step. In the 5th image is of Speed limit (20km/hr) but there's a bar above the sign which can be classified as No Vehicles sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set which was .

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model:

**Probability and Predictions**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yeild								            | 
| 1.0     				| Keep Right 								    |
| 1.0					| No Vehicles									|
| 0.5	      			| Ahead Only					 				|
| 1.0				    | Go straight or left      						|

**Output**

![alt text][image9]

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

