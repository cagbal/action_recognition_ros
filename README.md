# action_recognition_ros
ROS Package for recognizing actions of people

This repo is created to classify action classes defined in UCF101 dataset in an online manner. 

# Installation 

```sh
 git clone --recursive https://github.com/cagbal/action_recognition_ros.git
 cd action_recognition_ros/src/five_video_classification_methods/
 pip install -r requirements.txt
 cd <your_workspace> 
 catkin_make
 ```

# Pre trained weights 
Download them from: 
https://s3.eu-central-1.amazonaws.com/socratesweights/lstm-features.094-1.083.hdf5

And put it: 
<online_action_recognition_package_path>/src/five_video_classification_methods/data/checkpoints/lstm-features.094-1.083.hdf5

These weights are acquired through the training the network with UCF101 dataset. So, the action classes can be found: 
https://github.com/cagbal/action_recognition_ros/blob/master/src/action_classification/ucf_labels.py

# Run
Running is done via launch file. Please use the following command to launch the node: 
```sh
roslaunch action_recognition action_recognition.launch
 ```
This launch file can be found under launch folder and it uses action_recognition_params.yaml file to set parameters.



License
----

Apache License 2.0


# ***Special Thanks to https://github.com/harvitronix/five-video-classification-methods repo***



