# EE 286 Final Project: Music Source Seperation
Main Motivation/Reference: https://github.com/Kikyo-16/A-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis

## Quick Start
All commands are run once you change directory from the main project. 

### 1. Requirements.
Run the following command using pip3. This would install the necessary dependencies for this project.
```
pip3 install -r requirements.txt
```

### 2a. Data Preperation (Features)
Download the dataset from [URMP homepage](http://www2.ece.rochester.edu/projects/air/projects/URMP.html).After this, run the following command to generate your features, i.e., preprocess the dataset. Note that this might take a while, around 3 hrs. Assume that **ur_unzipped_dataset_folder** folder is the dataset you downloaded from the link and **dataset/hdf5s/urmp** is where the output features would be stored.
```
python3 src/dataset/urmp/urmp_feature.py --dataset_dir=ur_unzipped_dataset_folder --feature_dir=dataset/hdf5s/urmp --process_num=1 --sample_rate=16000 --n_fft=2048 --frames_per_second=100 --begin_note=21 --notes_num=88
```
### 2b. Data Preperation (Data)
Run the following command to create the data folder which would be used to train and test the network.
```
python3 src/dataset/urmp/urmp_generate_dataset.py --data=data/urmp --feature=dataset/hdf5s/urmp
```


### 3. Train the Network
Run the following command to train the network. 
```
python3 reyes_tan_models_upgraded.py --train_dir=train_results
```

### 4. Evaluate the Network
Run the following command to evaluate the network and produce the corresponding seperated music file. Note that the --epoch flag is to choose which epoch weights to evaluate. The --train_dir flag is where the training weights are stored.
```
python3 separate_test.py --train_dir=train_results --eval_dir=eval_results --epoch=0
```
