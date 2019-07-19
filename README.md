# Train-Predict-Landmarks-by-DAN
Train Predict Landmarks by Deep Alignment Network (DAN)

based on the code from:<br>
https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment  <br>
modified to our requirments.

# tested with: <br>
python 3.6 
tensorflow 1.12


# preparation
1- create folders named: <br>
"model_dir"<br>
"prepared_data_test"<br>
"prepared_data_train"<br>
"prepared_data_val"<br>
"results"<br>
"results_post"<br>

2- rearange your data such that for each image, you have a pts file contaring the locations of the landmarks. <br>
format of the pts file is same as the 300w face dataset.<br>
put your data of train, val and test of the folders "data_train/train", "data_train/val" and "data_test".<br>

following image shows an example of folder and files arragament after preparation:<br>

![Alt text](screen-16.15.58[19.07.2019].png?raw=true "Title")

# bacth file
now if you want you can use the .bat file for windows terminal which do all of the following parts automatically. <br>
Of course, you can write your own linux shell file. <br>


