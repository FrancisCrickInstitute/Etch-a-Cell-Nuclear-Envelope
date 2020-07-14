#!/usr/bin/env bash
###############################################################################
### This script checks to see if a keras model file called model.hdf5 exists
### inside "projects/nuclear/" and if it is missing, it downloads a trained
### model. After that the script runs the prediction pipeline through Docker
### on any image stacks in "projects/nuclear/resources/images/raw-stacks".
###############################################################################

# make sure we are in project dir
cd "$(dirname "$0")"
workingdir=$(pwd)

printf "> Checking Docker existence and version..."
docker --version &> /dev/null
if [[ ! $? -eq 0 ]]
then
  echo
  echo 'Please install Docker before using this script.'
  exit 1
fi

version=$(docker --version)
version=${version%%.*}
version=${version##Docker version }
if [[ $version -lt 19 ]]
then
  echo
  echo 'Docker version 19+ required.'
  exit 1
fi
echo 'DONE'

source_dir="projects/nuclear/resources/images/raw-stacks"
source_files=$(ls -1q $source_dir | wc -l)
if [[ source_files -eq 0 ]]
then
  echo
  echo 'No target image stacks found.'
  echo "To use this prediction script, place your image stacks in $source_dir"
  exit 1
fi

model_filename="projects/nuclear/model.hdf5"
if [[ ! -f $model_filename ]]
then
  echo "> No model found at $model_filename, downloading..."
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/22047516?private_link=5284fed9d683f4eaa0d5 -O $model_filename
else
  echo "> Using existing model found at $model_filename."
fi

echo '> Initiating docker and running prediction pipeline.'
docker build -t organelle-pipeline .
docker run --gpus all --entrypoint python3 -v "$workingdir/projects:/em/projects" organelle-pipeline -u run_prediction_pipeline.py --model $model_filename

docker_exit_code=$?

if [[ $docker_exit_code -eq 137 ]]
then
  echo
  echo '*** Docker ran out of memory ***'
  echo 'If you are using docker desktop, allocate more memory, otherwise run this script a higher spec machine.'
  exit 137
fi

output_dir="projects/nuclear/resources/images/scaled-predictions-stacks"
if [[ $docker_exit_code -eq 0 ]]
then
  echo
  echo "Predictions complete, the label stacks have been saved in $output_dir"
fi


