# Analysis and Prediction of Patient Mortality and Length of Stay

## Usage

### Data Download
Before extracting data, one need to request permission and download MIMIC III dataset from https://mimic.physionet.org/gettingstarted/access/ as csv.gz files

### Load data 
Before loading data, “LABEVENTS.csv.gz”, “CHARTEVENTS.csv.gz”, “OUTPUTEVENTS.csv.gz” should be saved in a folder named raw_data, and all loaded items would be saved in the raw_data folder.
```
python3 load_data.py 
```

### Data cleaning 
Before cleaning data, a folder named “remove_outlier” need to be created.
```
python3 clean_data.py 
```
### To review Exploratory Data Analysis of Length of Stay and Mortality
Please refer to the following jupyter notebook files 
* los.ipynb
* mortality.ipynb

### Extract Baseline Feature Set and Customized Feature Set
Please follow and run the folloing jupyter notebook files in order
* process_spas.ipynb
* process_data.ipynb
* feature_selection.ipynb

### Train MLNN for Short Stay/Long Stay/In-Hospital Mortality Prediction
Please follow the steps in the following jupyter notebook file
* NN.ipynb

### Running GRU Model
The GRU model needs to take a parameter --target to specify predicting length of stay or mortality rate. It could be run like:
```
python3 interpolation_GRU.py --target 3 
python3 interpolation_GRU.py --target 7
python3 interpolation_GRU.py --target m
```










