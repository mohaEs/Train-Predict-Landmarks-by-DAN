# Train-Predict-Landmarks-by-DAN
Train Predict Landmarks by Deep Alignment Network (DAN)

based on the code from:<br>
https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment  <br>
modified to our requirments.

# tested with: <br>
python 3.6 
tensorflow 1.12

# Paper:
This code is used for the following research. If you found it usefull, please cite the following document:

https://www.nature.com/articles/s41598-020-58103-6

@article{eslami2020automatic,
  title={Automatic vocal tract landmark localization from midsagittal MRI data},
  author={Eslami, Mohammad and Neuschaefer-Rube, Christiane and Serrurier, Antoine},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={1--13},
  year={2020},
  publisher={Nature Publishing Group}
}

Following repositories are also used for the mentioned paper:

https://github.com/mohaEs/Train-Predict-Landmarks-by-SFD

https://github.com/mohaEs/Train-Predict-Landmarks-by-MCAM

https://github.com/mohaEs/Train-Predict-Landmarks-by-Autoencoder

https://github.com/mohaEs/Train-Predict-Landmarks-by-dlib

https://github.com/mohaEs/Train-Predict-Landmarks-by-flat-net


# data preparation
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

# scaling and preparation
our input image sizes is 256x256 which should be changed to the size suitable for network.  <br>
> python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_train/train/ --output_dir ./prepared_data_train/  --img_size 112 <br>
> python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_train/val/ --output_dir ./prepared_data_val/  --img_size 112 <br>
> python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_test --output_dir ./prepared_data_test/  --img_size 112 <br>

# train
> python ./DAN_V2/DAN_V2_modified.py --dan_stage 1 --data_dir=./prepared_data_train/ --data_dir_test=./prepared_data_val/   --train_epochs=15  --num_lmark 68  --epochs_per_eval=1 -mode train --batch_size=40 <br>
> python ./DAN_V2/DAN_V2_modified.py --dan_stage 2 --data_dir=./prepared_data_train/ --data_dir_test=./prepared_data_val/  --train_epochs=45 --num_lmark 68 --epochs_per_eval=1 -mode train --batch_size=40 

# predict
the results will be saved in results folder.
> python ./DAN_V2/DAN_V2_modified.py -ds 2 --data_dir=./prepared_data_test/  --model_dir=./model_dir  --num_lmark 68 -mode predict <br>

# postprocess and rescale
the post processed results will be saved in results_post folder.
> python ./DAN_V2/postprocessing_nocrop.py --input_dir  ./results/ --output_dir ./results_post/  --img_size 256 <br>
<br>
<br>

![Alt text](results.png?raw=true "Title")
