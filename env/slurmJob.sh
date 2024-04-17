#!/bin/bash

#SBATCH --job-name=Vis
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --nodelist=nova[83]
#SBATCH --exclude=nova32
##SBATCH --nodelist=nova[82,83]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=main

##SBATCH --mem=50G
#SBATCH --mem-per-cpu=50G
#SBATCH --time=7-0

#SBATCH --mail-type=ALL




echo "---- env ! ----"

## ulimit -s unlimited
## ulimit -l unlimited
## ulimit -a

echo "------- setup done ! -----"
## Load the python interpreter
##clear the module
module purge


## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"




srun nvidia-smi

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRAINING iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

echo "--**Testing Gpu**--"
srun python gpu_availability.py
## srun nvidia-smi &&  python Train.py --config-file skydata_default_config.json
##srun nvidia-smi && python Train.py --config-file vedai_default_config.json
# # srun nvidia-smi && python3 Train.py --config-file googlemap_default_config.json 

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DATASETS iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# srun python scripts/check_dataset_integrity.py 
