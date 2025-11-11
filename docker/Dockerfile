FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive


RUN apt-get -o Acquire::https::dl.bintray.com::Verify-Peer=false -o Acquire::https::mirrors.tuna.tsinghua.edu.cn::Verify-Peer=false -o Acquire::https::mirrors.tuna.tsinghua.edu.cn::Verify-Host=false update
RUN apt-get update
RUN apt-get install -y unzip 
RUN apt-get install -y graphviz 
RUN apt-get install -y curl 
RUN apt-get install -y musescore3 
RUN apt-get install -y python3-pip 

RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt

# Hack to get around tensorflow-io issue - https://github.com/tensorflow/io/issues/1755
RUN pip install tensorflow-io
RUN pip uninstall -y tensorflow-io

COPY /notebooks/. /app/notebooks
COPY /scripts/. /app/scripts

ENV PYTHONPATH="${PYTHONPATH}:/app"