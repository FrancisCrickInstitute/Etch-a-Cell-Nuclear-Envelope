#!/usr/bin/env bash
###############################################################################
### This script downloads the files necessary to rerun training, in particular
### the reference EM image stacks and raw CSV file containing training labels.
### It then proceeds to run the full training pipeline through Docker.
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

printf "> Creating missing folders..."
if [[ ! -d "projects/nuclear/resources/" ]]
then
  mkdir "projects/nuclear/resources/"
fi

if [[ ! -d "projects/nuclear/resources/csv/" ]]
then
  mkdir "projects/nuclear/resources/csv/"
  mkdir "projects/nuclear/resources/csv/processed/"
fi

if [[ ! -d "projects/nuclear/resources/images/" ]]
then
  mkdir "projects/nuclear/resources/images/"
fi
echo 'DONE'

classificationsfilename="etch-a-cell-classifications-cleaned.csv"
# check if we already have the classifications csv
if [[ ! -f "projects/nuclear/resources/csv/$classificationsfilename" ]]
then
  echo '> Retrieving Zooniverse CSV data...'
  wget -q --show-progress https://ndownloader.figshare.com/files/21189879?private_link=48fa842e82d21702bf88 -O classifications.tar.gz
  tar -xvzf classifications.tar.gz -C "projects/nuclear/resources/csv/"
  rm classifications.tar.gz
else
  echo '> Zooniverse csv already exists, not redownloading.'
fi

if [[ ! -d "projects/nuclear/resources/images/raw-stacks/" ]]
then
  echo '> Downloading reference EM image stacks.'
  mkdir "projects/nuclear/resources/images/raw-stacks/"
  cd "projects/nuclear/resources/images/raw-stacks/"
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1416-1932-171.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1536-3456-213.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1584-6996-1.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1608-912-1.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1656-6756-329.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_1716-7800-517.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_2052-5784-112.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_2448-4704-271.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_2820-6780-468.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_2832-1692-1.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3000-3264-393.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3516-5712-314.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3576-5232-35.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3588-3972-1.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3624-2712-201.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3768-7248-143.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_3972-1956-438.tiff
  wget -q --show-progress --content-disposition ftp://ftp.ebi.ac.uk/empiar/world_availability/10478/data/ROIs/ROI_4320-1260-95.tiff
  cd "../../../../../"
else
  echo '> EM image stack folder already exists, not redownloading.'
fi


echo '> Initiating docker and running training pipeline.'
docker build -t organelle-pipeline .
docker run --gpus all --entrypoint python3 -v "$workingdir/projects:/em/projects" organelle-pipeline -u run_training_pipeline.py

if [[ $? -eq 137 ]]
then
  echo
  echo '*** Docker ran out of memory ***'
  echo 'If you are using docker desktop, allocate more memory, otherwise run this script on a higher spec machine.'
  exit 137
fi

