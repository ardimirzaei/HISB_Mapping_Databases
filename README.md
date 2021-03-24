# HISB_Mapping_Databases

While the files look like they are a mess, there is manner to the madness. 

# Step 1 - Create the Experiment Files.

The first file to run is the R file 'create_hpc_experiment_files.R'
These will build the python files for running an experiment, and will build the PBS files for running the experiments on the HPC. 
It will create a file called _EXPERIMENT_LIST.csv_, which contains a list of the experiemnts, the training and testing ranges, and the model used in that experiment. 

# Download Datasets

