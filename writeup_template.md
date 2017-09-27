## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image9]: ./examples/sliding_windows2.jpg
[image4]: ./examples/sliding_window.jpg
[image8]: ./examples/sliding_window2.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


The code to extract features is in features.py. In the iPython notebook I explicitly use the code under section "Visualize features".

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried to evaluate thoroughly different configuration choices. Please have a look at the parameter-tuning.ipynb.

In short summary I varied one parameter at a time, whereby I made exceptions where I felt like parameters are dependent of each other.

I ended up with a small feature vector and boosted baseline HoG accuracy from 92% to 96%. Combining with color and spatial features I achieve around 99%. However, the greatest puzzle is that my sliding window detector returns nonsense if I enable spatial features or color features. So I ended up using the HoG detector only. On top of that I believe that I have chosen the HoG classifier parameters in a way that does overfit to the training data and hence the generalisation performance in the video is rather bad.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

On the same data I compared training decision trees, linear SVM and nonlinear SVM in the parameter-tuning notebook under section "Compare classifier types". Linear SVMs outperform decision trees and nonlinear SVMs outperform linear SVMs. I evaluated my model choices using linear SVMs, but switched to nonlinear SVMs at the very end. In all cases I use sklearns train_test_split function and StandardScaler to center the features to zero mean and unit variance.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The Vehicle-Detection-Solution.ipynb has a section called "Sliding window object detector".

Since roughly the upper half of the image contains the horizon and sky, you can specify a tuple y_start_stop to be restrict the sliding window search to a range of y-values.

The sliding window searches line-by-line and shifts the window by an amount that is specified in terms of overlap. So an xy_overlap of 0.5 means that the window is shifted by half its size. The size of the sliding window xy_window is chosen to be 64x64. This actually ensures that no resizing needs to be done when classifying each individual patch. If we would have to rescale individual sliding window patches, that would increase the computational time greatly. Instead it makes sense to rescale the whole images just once before applying the sliding window search.

An example of a sliding window search at scale 1.0 can be found below:

![alt text][image3]

An example of a sliding window search at scale 2.0 can be found below:

![alt text][image9]

The maximum scale was chosen by looking at the test images. The test images contain a white car which is close to the car. Closer cars would appear the largest. Half of this white car is approximately 128x128 pixels large. So the classifier should be able to see the car if we support a scale of 2.0 (given that 64x64 is scale of 1.0).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I am searching over multiple scales and am painting heatmaps in the original image coordinate system. The resulting heatmaps are combined and a threshold of 2 is used to filter detections that have fewer responses.

Example bounding boxes based on unthresholded detections:

![alt text][image4]

After thresholding lower heatmap values the bounding boxes split or become smaller, even disappear (false positives).

![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap. I combine heatmaps over multiple scales and then thresholded that map with a value of 4 to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example image showing the detected (merged) bounding boxes, the underlying heatmap and the labeling image.

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I trained my classifier experimenting with various parameters to get a good feeling on how the classifier would behave. However, I think I made a mistake just having a rather small dataset and just a training and test split (vs. having a train/validation/test split). I seem to be optimizing a classifier that is overfitting the given small dataset and hence it doesn't generalize well.

I struggled to use the suggested pipeline where HoG features are subsampled. First I hit a blocker due to the fact that the implementation in the classroom contained some hard assumptions and didn't work with different configurations (hog_channel = 'ALL' or not using spatial features / histogram features).

I ultimately figured I could develop a basic multi-scale pipeline, which just about works. I am still not happy though with the large amount of false detections.

I think it would be very tedious to do a lot of manual data cleaning given the fact that we have already learned about overfitting, train/test/validation splitting, etc. in other courses.

While I appreciate the hands on experience with HoG, I do believe it would have been much more interesting to design this exercise in a way that we train a decent classifier based on convolutional neural nets. Especially since we learned about transfer learning it would have been really cool to train a state-of-the-art classifier or experiment with various other architectures. I felt like this project was a bit more focused on irrelevant boiler plate code.

On top of that my conclusion was that the color and the spatial features somehow ruin my pipeline and I can't really explain it. This was a really big blocker and I re-trained so many times when I was aware all the time that simple color features fail in typical real-world scenario. At least that's how I feel like.