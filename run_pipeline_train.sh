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
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237351?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237354?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237357?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237360?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237363?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237366?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21237714?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238497?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238953?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238956?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238962?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238989?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238992?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238995?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21238998?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21239001?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21239010?private_link=03f5e22febd6495904ba
  wget -q --show-progress --content-disposition https://ndownloader.figshare.com/files/21239016?private_link=03f5e22febd6495904ba
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

