** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on Nvidia's self-driving car model. It consists of a Normalization layer, next a cropping layer, 
followed by 5 convolution layers, one flatten layer, 3 fully connected layer & 1 output layer.
![Nvidia's self-driving car model][image1]

Layer 1: Image Normalization layer
Layer 2: Cropping Layer -- cropping 70 rows pixel from top of the image & 25 from bottom of the image
Layer 3: Convolution Layer -- 2x2 stride -- relu activation
Layer 4: Convolution Layer -- 2x2 stride -- relu activation
Layer 5: Convolution Layer -- 2x2 stride -- relu activation
Layer 6: Flatten layer
Layer 7: Fully connected layer
Layer 8: Fully connected layer
Layer 9: Fully connected layer
Layer 10: Output layer

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-52). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68). Tuned the no. of epochs to 3.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. There are around 40k training data.
I used a combination of center lane driving, recovering from the left and right sides of the road. 
The training data has a maximum of center lane driving & running through autonomous mode at locations where the car 
went out of track I trained for recovery modes for either shifting from left to right or right to left.
And a couple of laps was focused onto driving smoothly around the curves.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first network was a flattened image connected to a single output with the training data provided by Udacity. 
This kept my car moving in circle, getting inside the lake & coming out again & again :)
It uses Mean Square Error loss function to minimize the cost between the ground truth & the output predicted, running for 3 epochs.

Next added a Lambda layer for image normalization, which ensures the model will normalize input images when making predictions from drive.py 

The next change was to try LeNeT architecture with 5 epochs. This felt like a real driver :) 

Next after some improvements, tried Nvidia's autonomous driving network.  This was some real progress!
 
Improved this further by collecting more training data of multiple categories:
1. Driving in center
2. Driving at edges of the road
3. Recovering from edges to center
4. Driving in the counter-clockwise direction
5. Driving on the second track to generalize more
6. Driving focused on smooth curves

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					 | 
|:---------------------:|:----------------------------------------------:| 
| Input         		| 160x320x3 image   					         | 
| Normalization     	| Normalize the image 	                         |
| Cropping				| 70 rows pixel from top  & 25 pixel from bottom |
| Convolution 5x5	  	| 2x2 stride, 5x5 kernel, 24 depth               |
| Convolution 5x5	  	| 2x2 stride, 5x5 kernel, 36 depth               |
| Convolution 5x5	  	| 2x2 stride, 5x5 kernel, 48 depth               |
| Convolution 3x3	  	| 2x2 stride, 5x5 kernel, 64 depth               |
| Convolution 3x3	    | 2x2 stride, 5x5 kernel, 64 depth               |
| Flatten				|												 |
| Fully-connected       | 100 depth										 |
| Fully-connected       | 50 depth										 |
| Fully-connected       | 10 depth										 |
| Fully-connected       | 1 depth										 |
| Output                |                                                |

#### 3. Creation of the Training Set & Training Process


To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
