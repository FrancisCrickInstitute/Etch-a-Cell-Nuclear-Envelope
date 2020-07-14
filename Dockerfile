FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get -y upgrade && apt-get clean all
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y python3 python3-pip

COPY requirements.txt /em/requirements.txt
WORKDIR /em
RUN pip3 install -r requirements.txt

COPY . /em

ENTRYPOINT ["python3", "/em/run_prediction_pipeline.py"]
