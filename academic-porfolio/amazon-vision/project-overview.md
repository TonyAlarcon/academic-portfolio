# Amazon Vision Robotics 





## Dataset Description

### Traning Dataset
The database consists of 89 total images each containing 10 everyday non-perishable objects, whose image instances were taken from various angles. The raw unprocessed image database can be viewed here: https://drive.google.com/drive/folders/1qumej0KzYC_JoJAIZMYzu74lY1fFqhxI?usp=sharing. An example of the collected images, taken with an Iphone 11 Pro Max rear camara is shown below:


<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164947293-f5b69a55-cd3e-47fa-a8e1-1d44ca7dd52c.jpeg" width="150" height="200" title="Raw Image" alt="Employee data" />
 <img src="https://user-images.githubusercontent.com/37822940/164947365-46df7cd9-f864-41fd-abac-d8e38c7b8926.png" width="300" height="200" /> 

</p>
<p align = "center">
<b>(left) Raw Image (middle) Cropped Image (right) Image Mask </b>
</p>




### Synthetic Image Generator
These object images were ramdomly sampled and preprocessed to generate 44 synthetic images, each containing 8 objects in random positions and varying occlusion degree. For each object, the clutterizer autonomously cropped out, resized, obtained mask segmentation  and lastly placed them on a empty tote background image. These collection of 44 images were uttilized as the final training dataset, which can be viewed in the following link: (https://drive.google.com/drive/folders/1dPMrko20-cj9y4Ew-A7OeOrUTUDbmnmQ?usp=sharing)



<p align = "center">
 
 <img src = "https://user-images.githubusercontent.com/37822940/164947173-f80e486b-fbb3-4894-b3c1-b7eb8bbf1869.png">

</p>
<p align = "center">
<b>Synthetic Image Example</b>
</p>

### Testing Dataset 
The test set, consisted of 5 layouts of 9 images, each containing 10 objects randomly placed inside a cardboard box of dimension 24’’x16’’x11’’ in various confiduations of light, angle and occlussion degree. The entire test dataset can be viewed here (https://drive.google.com/drive/folders/1Yj7rvFcpE50ZK0-mEkr506RmMC9rOGcp?usp=sharing)


<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164946470-6458b7de-7bed-4d29-b2c3-70fe2ab87920.jpeg" align="center" width="200" />

</p>
<p align = "center">
 <b>Testing Image Instance</b>

</p>




There are various differences in the training set versus the testing set. The original obtained training set contained a single object per photo whose object to frame ratio was close to 100%. However, the testing dataset contains several objects in an image, whose object/frame pixel ratio was below 20%.  The proposed model utilizes a clutterizer script to create alternative synthetic training data to closely replicate the conditions in the testing data. Nevertheless, the objects in the synthetic image are scaled, positioned and rotated random positions which do not exactly match the test set. In addition, the testing set contains objects that were not included in the training set. Therefore, the model is able to generalize to distinct rotation angles, positions, brightness, and occlussion degrees in a decent manner. 

## Classification Accuracy
A classification accuracy achieved on the test set. Use the same metrics as in deliverable #3 (1 point)

### Training Dataset Accuracy

The following tables showcase the training accuracy results mAP (mean Average Precision) metric, a standard metric for object detection.
<div align="center">
 
|   AP   |  AP50  |  AP75  |  APs  |  APm  |  APl   |
|:------:|:------:|:------:|:-----:|:-----:|:------:|
| 60.457 | 76.676 | 72.612 |  nan  | 5.193 | 63.984 |

 </div> 
<p align = "center">
<b>Evaluation Results for Bbox</b>
</p>

<div align="center">
 
| category   | AP     |   
|:-----------|:-------|
| ball       | 78.839 |  
| cup        | 56.496 | 
| head set   | 48.827 | 
| tongs      | 38.412 |
| mouse      | 75.284 |
| hard drive | 67.674 |
| controller | 72.887 |
| keyboard   | 53.531 |
| hair brush | 53.259 |
| calculator | 59.366 |
 
  </div>
 <p align = "center">
<b>Per Cateory BBox AP</b>
</p>

<div align="center">
 
|   AP   |  AP50  |  AP75  |  APs  |  APm  |  APl   |
|:------:|:------:|:------:|:-----:|:-----:|:------:|
| 63.743 | 76.222 | 72.330 |  nan  | 4.901 | 67.616 |
 
  </div>
 <p align = "center">
<b>Evaluation Results for Segmentation</b>
</p>


<div align="center">
 
| category   | AP     | 
|:-----------|:-------|
| ball       | 88.464 | 
| cup        | 61.654 | 
| head set   | 49.661 | 
| tongs      | 42.418 |           
| mouse      | 80.623 |
| hard drive | 69.567 |
| controller | 74.922 |
| keyboard   | 51.074 |
| hair brush | 56.195 |
| calculator | 62.848 |
 
 </div>
 <p align = "center">
<b>Per Category Results for Segmentation</b>
</p>
 
### Testing Dataset Accuracy
The testing dataset accuracy was obtained manually by inspecting and enumerating the correct labels assigned during prediction. Correct labels here denotes the numbre of instances the model accurately assigned the correct category *and* segmentation/bbox to the object. This number was then divided by the total amount of objects per layout. 

1. Layout One Accuracy: 47.7%
 - Correctly Identified: 43
 - Total Amount of Objects: 90
 
2. Layout Two Accuracy: 34.7%
 - Correctly Identified: 31
 - Total Amount of Objects: 90 

3. Layout Three Accuracy: 62.22%
 - Correctly Identified: 56
 - Total Amount of Objects: 90

4. Layout Four Accuracy: 67.94%
 - Correctly Identified: 53
 - Total Amount of Objects: 78 (Here some objects were completely occluded and thus not counted in this total)

5. Layout Five Accuracy: 45.55% 
 - Correctly Identified: 41
 - Total Amount of Objects: 90

 

## Accuracy Discussion
Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates? (5 points)


### Incorrect Annotations of the HeadSet and Tongs Objects in Training Data
One of the issues presented in this model is the incorrect procedure for polygons annotations in the training set for the headset and tongs category. These two items are hollow at their center and the polygon annotations should have reflected this characteristic. Instead, the annotations indicate that their hollow centers are part of the object and as such the model is trained to think as such. The severity of this mistake is more apparent with the Headset Category where the pixel area of the hollow center is greater than the pixel area of the actual object. 

<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164985312-b148dd56-e5f9-4390-94ba-f7bfd86df86c.png"  title="Raw Image" alt="Employee data" />
 <img src="https://user-images.githubusercontent.com/37822940/164985875-6a68741e-5c67-4981-9486-c8f22de7377d.png" /> 

</p>
<p align = "center">
<b>Two Instances of the Headset Object </b>
</p>

While the model nevertheless was able to learn some features of the headset category, it could mostly make correct predictions in the test set when the headset was occluded. When the headset was completely non-occluded, the model expected different pixels at their hollow center and decided that it could not belong to this category. The images below illustrate this issue: 

<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164989855-be7a31da-05e5-4a53-961f-7a650fd5b158.png"  title="Raw Image" alt="Employee data" /> 

</p>
<p align = "center">
<b>Predictions on the Test Set When HeadtSet on not Occluded</b>
</p>

<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164990260-462cbb58-772e-4144-a220-28cee2f9adb7.png"  title="Raw Image" alt="Employee data" /> 

</p>
<p align = "center">
<b>Predictions on the Test Set When HeadtSet is occluded</b>
</p>

A similar problem is found for the tongs category. The image below shows the first step for image preprocessing:

<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164985963-3615eb11-4ac0-44ec-8cbc-57e4eb0a2331.png"  title="Raw Image" alt="Employee data" />
 <img src="https://user-images.githubusercontent.com/37822940/164985966-de0fc408-3a45-4ea3-a06b-fc15ee106fe7.png" /> 

</p>
<p align = "center">
<b>Two Instances of the Headset Object </b>
</p>


The below image shows the prediction on the tongs category when it is not occluded, or it's hollow center is shown.
<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164990441-f769ba4a-ac65-4cab-8fc5-40243ecb19ab.png"  title="Raw Image" alt="Employee data" /> 


</p>
<p align = "center">
<b>Predictions on the Test Set When Tongs is not Occluded</b>
</p>

<p align = "center">
 
<img src="https://user-images.githubusercontent.com/37822940/164990444-73312633-f626-4a42-bb30-c86c9e6cffea.png"  title="Raw Image" alt="Employee data" /> 

 The image below shows the prediction on the tongs category when the hollow center is not shown.

</p>
<p align = "center">
<b>Predictions on the Test Set When Tongs is occluded</b>
</p>

The solution to this is trivial, for items with hollow centers, we should be careful to annotate carefully. 


### Distinct Lighting Conditions Between Training and Test Data

While gathering the images for the set of objects, the only lighting condition similar to the testing set was ambient lighting. The items were not exposed to the "side light" or "top light" conditions which may have affected the results. It seems like the model was able to accurately find the objects (even occluded one) when the lighting condition was set to ambient in the testing data. For future work, I would be interested in investigating the impact in accuracy when including more diverse lighting condition in the training set, either by augmentation or real authentic photos. Additionally, I would like to implement a version of the GradCAM in order to verify the regions of interest that helps the algorithm makes it's decision and study the effects in lighting conditions in that specific region. 
