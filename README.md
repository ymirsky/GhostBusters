# GhostBusters

In this repository you will find a Python Keras implementation of GhostBusters, including download links to the pre-trained models and datasets. The model is based on the following publication:

*Ben Nassi, Yisroel Mirsky, Dudi Nassi, Raz Ben-Netanel, Oleg Drokin, Yuval Elovici. Phantom of the ADAS: Securing Advanced Driver-Assistance Systems from Split-Second Phantom Attacks. The 27th ACM Conference on Computer and Communications Security (CCS) 2020*

GhostBusters is a proposed countermeasure against Phantom attacks on driverless vehicles and advanced driver assist systems (ADAS). A Phantom attack is where an object is projected or digitally displayed for a split-second near a vehicle causing the vehicle to behave unpredictably. For example, the projection of a person on the road can trigger the collision avoidance system, causing the car to stop or swerve in a dangerous manner. Another example is where a false road sign is projected on a wall nearby which alters the car's speed limit.

*The Attack Model:*
![](https://github.com/ymirsky/GhostBusters/raw/master/attack_model.png)


This attack raises great concern, because unskilled attackers can use split-second phantom attacks against ADASs with little fear of getting caught, because  
1. there is no need to physically approach the attack scene (a drone can project the image or a digital billboard can display it),
2. there will be no identifying physical evidence left behind, 
3. there will likely be few witnesses, since the attack is so brief, and 
4. it is unlikely that the target of the attack will try to prevent the attack (by taking control of the vehicle), since he/she won't notice anything out of the ordinary.

To counter this threat, we propose **GhostBusters**: a committee of machine learning models which validates objects detected by the on-board object detector. 
The GhostBusters can be deployed on existing ADASs without the need for additional sensors and does not require any changes to be made to existing road infrastructure. It consists of four lightweight deep CNNs which assess the realism and authenticity of an object by examining the object's reflected light, context, surface, and depth. A fifth model uses the four models' embeddings to identify phantom objects. Through an ablation study we have found that by separating the aspects our solution is less reliant on specific features. This makes it more resilient than the baseline model and robust against adversarial attacks. For example, it is hard to make a physical adversarial sample on optical flow.


## Who are the GhostBusters?
Our committee of experts, called the GhostBusters, consists of four deep CNN models, each focusing on a different aspect (see the architecture figure below). The models receive a cropped image of a traffic sign (x^t) and then judge if the sign is authentic and contextually makes sense:

* **Context Model.** This CNN receives the context: the area surrounding the traffic sign. This is obtained by taking x^t, re-scaling it to a 128x128 image, and then setting the center (a 45x45 box) to zero. Given a context, the model is trained to predict whether a sign is appropriate or not. The goal of this model is to determine whether the placement of a sign makes sense in a given location.

* **Surface Model.** This CNN receives the sign's surface: the cropped sign alone in full color. This is obtained by cropping x^t and re-scaling it to a 128x128 image. Given a surface, the model is trained to predict whether or not the sign's surface is realistic. For example, a sign with tree leaves or brick patterns inside is not realistic, but a smooth one is.

* **Light Model.** This CNN receives the light intensity of the sign. This is created with the cropped and scaled x^t, by taking the maximum value from each pixel's RGB values (x[i,j] = arg max k: x^t[i,j,k] ). The goal of this model is to detect whether a sign's lighting is irregular. This can be used to differentiate real signs from phantom signs, because the paint on signs reflects light differently than the way light is emitted from projected signs.

* **Depth Model.** This CNN receives the apparent depth (distance) of the scenery in the image. To obtain this, we compute the optical flow between x^t and the same space from the last frame, denoted x^{t-1}. The optical flow is a 2D vector field where each vector captures the displacement of the pixels from x^{t-1} to x^t. In particular, with OpenCV, we utilize the Gunner Farneback algorithm to obtain the 2D field v, and then convert it to an HSV image by computing each vector's angle and magnitude and then by converting it to an RGB image before passing it to the CNN. The significance of this approach is that we can obtain an implicit 3D view of the scenery while the vehicle is in motion. This enables the model to perceive the sign's placement and shape better using only one camera. 

To make a prediction on whether or not a sign is real or fake, we combine the knowledge of the four models into a final prediction: As an image is passed through each of the models, we capture the activation of the fifth layer's neurons. This vector provides a latent representation (embedding) of the model's reasoning on why it thinks the given instance should be predicted as a certain class. We then concatenate the embeddings to form a summary of the given image. Finally, a fifth neural network is trained to classify the cropped sign as real or fake using the concatenated embeddings as its input. 

*The Architecture: Meet the GhostBusters*
![](https://github.com/ymirsky/GhostBusters/raw/master/architecture.png)



## This Version's Features and Limitations

**Features**
* Phantom detection (road signs only)
* Dataset creator (input full frame videos or images and the road signs are automatically extracted) 
* Pre-trained models and datasets
* Command Line Interface (CLI) tool 
* Example python scripts for customization and quick-start use in your own projects

**Limitations**
* The models were Trained using footage from a Xiaomi dashcam. Therefore, the pre-trained model *may not work on your own footage* unless you use the same camera (due to distortions). Please re-train the model on your own fottage using the provided dataset creator tools. 
* This version selects the best model based on validation loss during training (not AUC like in the paper). This may influence the results.


# The Code
The code has been written with OOP in mind. This repo contains example scripts for performing every step of GhostBusters, and the primary source code [GB_model.py](GB_model.py). The code is organized as follows:

An all-in-one tool:
* **[GhostBusters.py](GhostBusters.py)**	: a CLI tool for processing datasets, training models, and making predictions. See below for more details.

Example scripts for using the GhostBusters in Python:
* **[1-preprocess_dataset.py](1-preprocess_dataset.py)**	: An example script for using the dataset creator. The code (1) takes a video, (2) detects all road signs in each frame using a pre-trained [model](https://github.com/aarcosg/traffic-sign-detection) (you must download the model using the link below), (3) extracts four versions of each road sign (one for each expert) and then saves them to a directory.
* **[2-train_GB.py](2-train_GB.py)**	: An example script for training the GhostBusters on a preprocessed dataset. You must provide *three* separate preprocessed datasets (folders): one based on real signs, one based on fake signs (phantoms), and one with no signs in it (extracted automatically be the dataset creator when processing a 'real sign' dataset.
* **[3-execute_GB.py](3-execute_GB.py)**	: An example script for using the GhostBusters to make predictions on a preprocessed dataset. 

Source code files (the implementation):
* **[GB_extractor.py](GB_extractor.py)**	: The class for making preprocessed datasets from videos or images (videos are required for the depth [optical] expert). The extractor has a dependency on the Tensorflow Object Detection API which has a subdependency on the protobuf library (see below for details). To use it, you also need the pre-trained model (see below).
* **[GB_model.py](GB_model.py)**			: The class for training a Ghostbuster model using three preprocessed datasets (folders): real, fake, nosign.  
* **[utils.py](utils.py)**			: Helper functions used by the extractor and training classes. 


## Installation 
Implementation nodes:
* Tested on an Ubuntu System with 256GB RAM, Xeon E7 CPUs: using one Nvidia Titan RTX (24GB), Driver 450.51.05, CUDA Version 11.0.
* Tested using Anaconda 3.8.5, Keras with the tensorflow back-end v2.3, and v2.2 GPU (see [environment.yaml](environment.yaml))

### Installation for Model Training and Execution ONLY
1. Download the pre-trained models [from here](https://drive.google.com/file/d/1hAMOT0Nxe9MMhkY8w4_bY0Xz9WplcyNP/view?usp=sharing) and put them into the 'models' directory.
2. If you want to train/execute the model on our datasets, then download them [from here](https://drive.google.com/file/d/1bi0sklYmd83i6qcBQ45VNSdKqYio4r1k/view?usp=sharing) and put them into the 'data' directory.
3. Install the dependencies:
```
pip install --upgrade tensorflow keras pandas
```

### Installation for using the Dataset Creator
For creating a new dataset, you will need access to the TensorFlow Object Detection API, Protobuf, and OpenCV
1. Download the pre-trained object detection model (faster_rcnn_inception_resnet_v2_atrous) [from here](https://drive.google.com/file/d/1mp98ICdmvsdF_ys12pKLJjjE2u7VilNa/view?usp=sharing) and put it in the models directory ([original source](https://github.com/aarcosg/traffic-sign-detection)).
2. Install the Tensorflow Object Detection API ([complete instructions here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)):
	1. Install Google Protobuf:
		1. Run: ```sudo apt-get install autoconf automake libtool curl make g++ unzip```
		2. From https://github.com/protocolbuffers/protobuf/releases, download the protobuf-all-[VERSION].tar.gz.
		3. Extract the contents and cd into the directory
		4. Run:
		```
		./configure
		make
		make check
		sudo make install
		sudo ldconfig # refresh shared library cache.
		```
		5. Check if it works: ```protoc --version```
	2. Install the Detection API:
		1. Download the API to a directory of your choice: ```git clone https://github.com/tensorflow/models.git```
		2. cd into the TensorFlow/models/research/ directory
		3. Run: 
		```cp object_detection/packages/tf2/setup.py .
		python -m pip install .
		```
	3. Install OpenCV
		1. Run: ```pip install opencv-python```

### Conflicts, Missing Modules, and Errors... Oh My!
If you are having trouble running the code after following the above steps (e.g., missing modules or version conflicts), you can install the Anaconda virtual environment we used:

cd into the GhostBuster directory and run:
```
conda deactivate
conda env create --file environment.yaml
conda activate ghostbusters
```


## Using the CLI Tool
We will now review how to use the example scripts provided in this repo. To get general help on the tool, run:
```
$ python GhostBusters.py -h

usage: GhostBusters.py [-h] [-pr <vid_path>] [-pf <vid_path>] [-t] [-e <data_dir>] [-dd <data_dir>] [-md <model_dir>] [--exp_epochs E] [--com_epochs E] [--device_id ID] [--pred_path <pred_dir>]

A CLI tool for GhostBusters, for preparing datasets, training, and execution (prediction) For more information, please see our paper: Ben Nassi, Yisroel Mirsky, ... Phantom of the ADAS: Securing Advanced
Driver-Assistance Systems from Split-Second Phantom Attacks, CCS 2021. Tool developed by Yisroel Mirsky.

optional arguments:
  -h, --help            show this help message and exit

Procedures:
  Use one or more of these flags

  -pr <vid_path>, --proc_real <vid_path>
                        Extract a real-sign dataset from the given video. Will automatically extract nosign examples. E.g., $ python GhostBusters.py -pr /home/user/real_vid.mp4 (default: None)
  -pf <vid_path>, --proc_fake <vid_path>
                        Extract a fake-sign dataset from the given video (default: None)
  -t, --train           Extract a dataset from a video: The given path (default: False)
  -e <data_dir>, --predict <data_dir>
                        Predict on the given path. The path must be to a directory containing four subdirectories of the preprocessed signs (context, surface, light, optical). Outputs two columns: col0 is
                        probability of being real, col1 is probability of being fake. (default: None)

Parameters:
  Configurable parameters (optional)

  -dd <data_dir>, --data_dir <data_dir>
                        Set the save/load data directory. (default: data/)
  -md <model_dir>, --model_dir <model_dir>
                        Set the save/load model directory. (default: models/)
  --exp_epochs E        Training eopochs for each expert. (default: 25)
  --com_epochs E        Training eopochs for the combiner. (default: 10)
  --device_id ID        The ID of the GPU to use: 0, 1, ... or "" for CPU. (default: 0)
  --pred_path <pred_dir>
                        Set the save path for the predictions, saved as a csv. (default: predictions/pred.csv)
```

### Step 1: Build a Training/Test Set (or use our dataset and skip to step 2)
To train the GhostBusters model, you will need three preprocessed datasets (real signs, fake signs, and nosigns). Videos taken at night so the distribution will fit the phantoms as well. The video should be taken from driver's point of view (front of car) and can be recorded while casually driving around. 

To extract real signs from a video, run
```
$ python GhostBusters.py -pr <vid_path>
```
where ```<vid_path>``` is the the path to your video. Under the directory called 'data', two subdirectories will be made: 'real' containing the processed road signs and 'real_nosign' containing examples with no signs for training the context expert. You can change the data directory by using the ```-dd <data_dir>``` flag.

Similarly, to extract fake signs from a video, run
```
$ python GhostBusters.py -pf <vid_path>
```
The processed signs will be saved to 'data/fake/' unless the ```-dd <data_dir>``` flag is used, where <data_dir> is the alternate directory path.

You can run these commands multiple times on different videos since the data files are saved with unique filenames based on a hash of the input video name.

### Step 2: Train the Model (or use our pre-trained model and skip to step 3)
To train the model on the processed datasets, run
```
$ python GhostBusters.py -t
```
By default, the tool will use the data stored in 'data/real', 'data/real_nosign', 'data/fake'. To redirect the trainer to a different dataset location, use the ```-dd <data_dir>``` flag. There must be three directories in <data_dir> labeled 'real', 'real_nosign', and 'fake'.

You can also configure the number of epochs used to train each of the experts (```--exp_epochs```) and the combiner model (```--com_epochs```). You can also specify which GPU to use with the ```--device_id``` flag.

### Step 3: Execute the Model on a preprocessed dataset
To use a model to make predictions on a processed dataset, run
```
$ python GhostBusters.py -e <data_dir>
```
where <data_dir> is the directory containing a single preprocessed dataset (e.g., 'data/fake'). The results will be saved to a csv in a 'data/predictions' directory where column 0 is the probability of the sign being real, column 1 is the probability of the sign being fake, and column 2 is the filename associated with that sample (taken from the context expert's subdirectory to save space since it should match the other experts).



# Datasets & Models
* [Pre-processed Dataset](https://drive.google.com/file/d/1bi0sklYmd83i6qcBQ45VNSdKqYio4r1k/view?usp=sharing)
* [Pre-trained GhostBuster Models](https://drive.google.com/file/d/1nhqo9cYH3QZSzn3kCIv9JmL6tdeW3Or5/view?usp=sharing)
* [Pre-trained Roadsign Detector Model](https://drive.google.com/file/d/1mp98ICdmvsdF_ys12pKLJjjE2u7VilNa/view?usp=sharing)


# License
See the [LICENSE](LICENSE) file for details


# Citations
If you use the source code in any way, please cite:

*Yisroel Mirsky, Tom Mahler, Ilan Shelef, and Yuval Elovici. 28th USENIX Security Symposium (USENIX Security 19)*
```
@inproceedings {ghostbusters@2020,
author = {Ben Nassi and Yisroel Mirsky and Dudi Nassi and Raz Ben-Netanel and Oleg Drokin and Yuval Elovici},
title = {Phantom of the ADAS: Securing Advanced Driver-Assistance Systems from Split-Second Phantom Attacks},
booktitle = {The 27th ACM Conference on Computer and Communications Security (CCS)},
year = {2020},
publisher = {ACM}
}
```

Contact:

Yisroel Mirsky
yisroel@post.bgu.ac.il
