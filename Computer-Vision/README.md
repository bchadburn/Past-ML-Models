# Indoor-Outdoor Image Classification

![alt text](https://www.csun.edu/sites/default/files/AS-Earth_Month-Outdoor_Online.jpg)

## Getting Started
Download images, video_category_data.json, and vocabulary.csv

2 options: 
1. Download subset of images from an AZURE storage container, video_category_data.json and vocabulary.csv
https://dresearch.blob.core.windows.net/public-indoor-outdoor/indoor_outdoor.zip
2. Skip create_data_set step: Download file "indoor_outdoor_images". This folder contains 
169 curated images. 
   - The images were selected from the AZURE subset of images using the following classes
     - indoor_scenes = {'Bedroom', 'Bathroom', 'Classroom', 'Office', 'Living Room', 'Dining Room', 'Room'
     - outdoor_scenes = {'Landscape', 'Skyscraper', 'Mountain', 'Beach', 'Ocean'}
   - Images that were incorrectly labelled as outdoor, indoor were removed. No hard/easy images were removed or added
    
**1st method**: Ensure parent folder "indoor_outdoor" is in project director. 
Within this folder include a folder "images". Also, within the parent folder include the two files: video_category_data.json and vocabulary.csv
If using this method, after creating dataset, you may want to review images as some
completely blank and several are mislabelled. These curated images are provided in the zip file "indoor_outddoor_images."

**2nd method**: Unzip and place folder "indoor_outdoor_images" in project directory and skip step: Creating Data Set.

Along with the image related files, ensure you have the following scripts in project director.
* create_data_set.py
* training.py
* image_processing_unit_test.py
* single_image_predictions.py
* Utils folder with preprocess_image.py and utils.py 
* Model folder with model.py
* requirements.txt file
* environment.yml (for conda users)

## Installation

- Create a python virtual environment in the system and activate it.

**Installation using pip:**
  - `pip install virtualenv`
  - `virtualenv <env_name>`
  - `source <env_name>/bin/activate`

Install the dependencies for the project using the requirements.txt
  - `pip install -r requirements.txt`

**Installation for conda users:**

The packages may fail to load if using installing from requirements.txt file as conda-forge may be required to download certain packages.
Instead, use environment.yml file.
- conda env create --name <env_name> --file=environment.yml
- conda activate <env_name>
    
### Creating data set
Run python create_data_set.py from CLI. The default destination of images is
"indoor_outdoor_images". 

To pass a different path run: python create_data_set.py --image_destination <path_to_folder>
If you pass a different path, you will need to pass the new image source path when running
training.py (see [Training section](#Training) for details).

As long as original images are in proper folder the script will move relevant images to new folder for training.

#### Description: 
Script uses video_category_data.json to map images to specific labels. It uses vocabulary.csv
to then map the labels to 'class' names listed in the variables indoor_scenes and outdoor_scenes. 
Relevant images are then moved to a new folder and are given a prefix to indicate whether the image should 
be indoor or outdoor (i.e. "0-" to specify an indoor image, "1-" for outdoor).

##### important variables
Modify directories, classes, sub-classes (e.g. for indoor, images labeled as "Bedroom", "Bathroom" etc. are 
all categorized as "indoor"), or class labels:
* image_source = 'indoor_outdoor/images' 
* image_dest = 'indoor_outdoor_images' 
* indoor_scenes = ('Bedroom', 'Bathroom', 'Classroom', 'Office', 'Living Room', 'Dining Room', 'Room')
* outdoor_scenes = ('Landscape', 'Skyscraper', 'Mountain', 'Beach', 'Ocean')
* indoor_label = 0
* outdoor_label = 1

##### Important functions
* map_classes: Maps Indoor, Outdoor to relevant 'class' labels. If other classes or wanted, 
  the function will need to be modified.
* map_parent_category: maps images to indoor, outdoor classes. If other classes or wanted, 
  the function will need to be modified.

### Training
Run python training.py

The default source of training images is indoor_outdoor_images. To specify source run:
python training.py --dataset <path_to_folder>
Other optional parameters can be passed for model training including:
- --epochs:  The number of epochs that will be used to train the initial classification model.
- --learning_rate: The learning rate that will be used to train the model
- --fine_tuning_epochs: The number of epochs that will be used to fine tune the model. If zero is specified, the model will not 
                                go through the fine-tuning process
- --fine_tuning_learning_rate:  The learning rate that will be used to fine tune the model.   

#### Description
Script trains a CNN model by adding layers on top of a pre-trained ResNet model. During fine-tuning stage, it unfreezes all layers and finishes training model.
Outputs confusion matrix, and model predictions and confidence scores for specific images. Returns overall model 
performance on validation dataset, and lastly outputs trained model.

Troubleshooting note: To run script, ensure system is using numpy version specified in requirements.txt file for numpy version. 
Script can throw the error "NotImplementedError: Cannot convert a symbolic Tensor..." for some newer versions of numpy.

### Making single image predictions
Run python single_image_predictions -i {image path}  
For example, if passing images used for train/val: python single_image_prediction.py -i indoor_outdoor_images/0-_2hRjVpJtdY.jpg

#### Description
Loads model and makes prediction on provided image. Prints predicted class and confidence score.
 
### Run Tensorflow GPU vs CPU test
When running training.py file, the script will run a tensor through a simple CNN layer 
through CPU and will try using GPU and compare test times. The results will be printed in the console. 

If tensorflow-gpu is correctly configured, the GPU should be considerably faster, 
although its dependent on GPU. Using NVIDIA GeForce RTX 2080 Super, the GPU speed over CPU is 600+% for this test.

### Run Image Unit Test
Run python image_processing_unit_test.py -i {image path}

#### Description
Script is meant to review output provided by tf.io.read_file and tf.image.decode_jpeg
and compared output to PIl and CV2 image processing results. Image values are compared to identify differences, check if differences 
are minor. I used this to check if tf image values are comparable and whether PIL and CV2 can also be used to process images
for predictions. This is particularly helpful when serving a model and considering whether to use other options for loading images to make predictions. 

Results: I found while PIL and CV2 image values were identical, there were minor differences with tf method for loading and decoding. Found out the default TF decode method 
"INTEGER_FAST" speeds up processing but doesn't match other methods. Tried "INTEGER_ACCURATE" but found same results. Found out PIL doesn't provide integer accurate decompression. OpenCV on the other hand does 
and before TF 2.0 the image values are identical if TF method "INTEGER_ACCURATE" is passed. However, I was not able to reproduce using tf 2.0 with either method.
Verified image values were very similar. Even still, unit tests showed predictions differed widely when using the other processing methods.

### Model Improvements

**Correcting Image Labels:** Initially 183 indoor and outdoor images were found and moved to the training folder. The images
were reviewed to ensure labels were correctly assigned. 14 images were deleted as they were bad images (some blank, some were irrelevant). A few images 
labelled as indoor were outdoor images and visa-versa, so the labels were changed accordingly. 

**Model Performance:** After curation, the model's performance fluctuates between 95-100%. 
Small model architecture and parameter tweaking were needed to improve from 94% to high 90s but otherwise, the model has high performance 
even if the top layers are changed a bit. However,
the results can vary a few percent between trainings. 

**Next Steps:** To further improve the solution, I probably would enquire about the business requirement for accuracy, reliability, and speed and whether more improvement is needed. Typically, a final test set would also need to be created from images
we haven't seen and that properly reflect images the model would see in production.

**Model Improvements:** Otherwise, to improve the model further I would focus on a 'data centric' approach. 

1. Perform another round of curation and make sure no other bad images were missed. 

2. Identify the type of mistakes the model is making and how that differs
between trainings. For that reason, predictions for each image were exported after training.
   
3. Add other indoor/outdoor categories (similar to building, house etc) where images could be added or find other options to add more data. 
   It's likely to have a more reliable model, more images and examples would be useful given the high variability between images and low validation count.
   
4. Review data augmentations: With more time. I would be helpful to review different data augmentation methods and values. Given the 
   variability in our images, more aggressive augmentation could be added.

5. Hyper-parameter tuning, and fiddling with different architectures could also be useful, but I'd first start with 
business requirements and understanding the current limitations of the existing model. 
If model improvements are warranted, I'd probably use Keras Tuner to help efficiently explore the hyper-parameter search space for model parameters
and augmentation and fine-tune parameters from there. Finally, stacked and/or ensemble methods could be used to try getting optimum performance.

Speed: Given the low number of images, the model training takes place in under a minute. Since we are using gpu enabled tf, tf.data instead of slower methods like
Keras generators, parallel processing for image processing and loading, prefetch images in CPU while GPU runs etc. 
the model training is capable of training efficiently on a much larger dataset. 