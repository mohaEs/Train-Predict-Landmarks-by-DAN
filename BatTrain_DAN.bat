python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_train/train/ --output_dir ./prepared_data_train/  --img_size 112
python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_train/val/ --output_dir ./prepared_data_val/  --img_size 112
python ./DAN_V2/preprocessing_nocrop.py --input_dir  ./data_test --output_dir ./prepared_data_test/  --img_size 112

python ./DAN_V2/DAN_V2_modified.py --dan_stage 1 --data_dir=./prepared_data_train/ --data_dir_test=./prepared_data_val/   --train_epochs=15  --num_lmark 68  --epochs_per_eval=1 -mode train --batch_size=40
python ./DAN_V2/DAN_V2_modified.py --dan_stage 2 --data_dir=./prepared_data_train/ --data_dir_test=./prepared_data_val/  --train_epochs=45 --num_lmark 68 --epochs_per_eval=1 -mode train --batch_size=40

python ./DAN_V2/DAN_V2_modified.py -ds 2 --data_dir=./prepared_data_test/  --model_dir=./model_dir  --num_lmark 68 -mode predict
python ./DAN_V2/postprocessing_nocrop.py --input_dir  ./results/ --output_dir ./results_post/  --img_size 256