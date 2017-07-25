** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_model]: ./examples/nVidia_model.png "Model Visualization"
[recovery_image]: ./examples/recovery_image.jpg "Recovery Image"
[recovery_image2]: ./examples/recovery_image2.jpg "Recovery Image 2"
[normal_image]: ./examples/normal_image.jpg "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md  summarizing the results
* video.mp4 the output video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

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

My model is based on Nvidia's self-driving car model. 
![Nvidia's self-driving car model][nvidia_model]

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

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49-52). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 59). Tuned the no. of epochs to 3.

Training data was chosen to keep the vehicle driving on the road. There are around 40k training data.
I used a combination of center lane driving, recovering from the left and right sides of the road. 
The training data has a maximum of center lane driving & running through autonomous mode at locations where the car 
went out of track I trained for recovery modes for either shifting from left to right or right to left.
And a couple of laps was focused onto driving smoothly around the curves.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![center driving][normal_image]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.
These image show what a recovery looks like starting from :

![alt text][recovery_image2]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had around 40k images.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 3 & I used an adam optimizer so that manually training the learning rate wasn't necessary.
