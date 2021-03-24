# HISB_Mapping_Databases

While the files look like they are a mess, there is manner to the madness. 

## Step 1 - Create the Experiment Files.

The first file to run is the R file 'create_hpc_experiment_files.R'
These will build the python files for running an experiment, and will build the PBS files for running the experiments on the HPC. 
It will create a file called _EXPERIMENT_LIST.csv_, which contains a list of the experiemnts, the training and testing ranges, and the model used in that experiment. 

## Step 2 - Download Datasets

You can download all the dataset from good as long as you have the google sheet id. You'll need a token.pickle and credntials.json to access GoogleSheet. 

## Step 3 - Run Experiment on the HPC

The resultant PBS files can be submitted to the HPC queue for processing

### Optional extra - run an individual file.

The __pyfile__01_build_models is the temptlate PY file. You can adjust TRAINING_RANGES, PREDICTION_RANGES, MODE_NAME, ANALYSIS_METHOD from lines 53-56 to have a custom and specific file to run. 

